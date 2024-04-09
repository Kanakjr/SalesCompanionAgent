# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from skimage.measure import label, regionprops
from copy import deepcopy
import os
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    Float,
    MetaData,
    Table,
    inspect,
    select,
    func,
    Numeric,
    FLOAT,
    INTEGER,
)
import re
from langchain_community.utilities.sql_database import SQLDatabase
from typing import Any, List, Tuple, Dict

from random import sample

import warnings

warnings.filterwarnings("ignore")


def drop_byte_columns(df):
    # get list of columns with byte type
    byte_cols = [
        col
        for col in df.columns
        if df[col].dtype == "object"
        and df[col].size > 0
        and isinstance(df[col][0], bytes)
    ]
    # drop byte columns
    if byte_cols:
        df = df.drop(columns=byte_cols)
    return df


class StructuredDatabase(SQLDatabase):

    schema_str: str = None

    @classmethod
    def from_excel_or_csv(cls, filepath, sqlite_in_memory=True):
        basename = os.path.basename(filepath)
        basename_wo_extention, file_extention = os.path.splitext(basename)
        filepath_wo_extention = os.path.splitext(filepath)[0]

        if not os.path.exists(filepath):
            raise Exception(f"{filepath} do not exist")

        # load data from file
        if file_extention == ".xlsx":
            try:
                tables = pd.read_excel(filepath, sheet_name=None, header=None)
            except:
                raise Exception(f"{filepath} is corrupt")

            for tablename, table in list(tables.items()):
                sheet_tables = cls._get_tables_using_skimage(table)
                sheet_tables = sorted(
                    sheet_tables,
                    key=lambda table: table.shape[0] * table.shape[1],
                    reverse=True,
                )
                if len(sheet_tables) >= 1:
                    table = sheet_tables[0]
                    table = cls._add_header_to_table(table, 1)
                    tables[tablename] = table

        elif file_extention == ".csv":
            try:
                table = pd.read_csv(filepath)
            except:
                Exception(f"{filepath} is corrupt")
            tables = {}
            if len(table) >= 1:
                tables[basename_wo_extention] = table

        # clean data suitable for sql and gen ai and correct the data types
        tables_cleaned = {}
        for tablename, table in tables.items():
            table = table.loc[:, ~table.columns.str.lower().duplicated()]
            table.columns = table.columns.str.lower()
            table.columns = table.columns.str.strip()
            table.columns = table.columns.str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)

            for column in table.columns:
                try:
                    table[column] = pd.to_numeric(table[column])
                except:
                    pass

            tablename_new = re.sub(r"[^a-zA-Z0-9_]", "_", tablename)
            tables_cleaned[tablename_new] = table

        # Populate SQLite with data from dictionary and store metadata
        if sqlite_in_memory:
            database_url = "sqlite:///:memory:"
        else:
            database_url = f'sqlite:///{filepath_wo_extention + ".db"}'
            if os.path.exists(filepath_wo_extention + ".db"):
                os.remove(filepath_wo_extention + ".db")

        engine = create_engine(database_url)
        metadata = MetaData()
        for tablename, table in tables_cleaned.items():
            # Create table
            cols = []

            for column in table.columns:
                s = table[column]
                s_dtype = s.dtype.kind
                if s_dtype == "i":
                    sqlalchemy_dtype_object = Integer
                elif s_dtype == "f":
                    sqlalchemy_dtype_object = Float
                elif s_dtype == "b":
                    sqlalchemy_dtype_object = Boolean
                elif s_dtype == "O":
                    sqlalchemy_dtype_object = String
                else:
                    raise Exception()
                is_unique = s.is_unique
                cols.append(Column(column, sqlalchemy_dtype_object, unique=is_unique))

            sqlalchemy_table = Table(tablename, metadata, *cols)
            metadata.create_all(engine)

            # Insert data
            # table.to_sql(tablename, engine, if_exists='replace', index=False)
            with engine.connect() as conn:
                conn.execute(sqlalchemy_table.insert(), table.to_dict(orient="records"))

        return cls(engine)

    def df_from_sql_query(self, query):
        return pd.read_sql(query, self._engine)

    def fetch_distinct_row_values(
        self, column_list: List[Tuple[str, str]], num_max_distinct=12
    ) -> Dict[Tuple[str, str], List[Any]]:
        column_list = eval(column_list)
        column_values_dict = {}
        with self._engine.connect() as connection:
            for table_name, column_name in column_list:
                # Ensure the table and column exist
                if table_name not in self._metadata.tables:
                    raise ValueError(f"Table {table_name} not found in the database")
                table = self._metadata.tables[table_name]
                if column_name not in table.columns:
                    raise ValueError(
                        f"Column {column_name} not found in table {table_name}"
                    )

                query = select(table.columns[column_name]).distinct()
                distinct_values = connection.execute(query).fetchall()
                distinct_values = [
                    item[0] for item in distinct_values
                ]  # Flatten the list
                # If numeric column and distinct values greater than num_max_distinct, fetch min and max
                if (
                    isinstance(table.columns[column_name].type, Numeric)
                    or isinstance(table.columns[column_name].type, INTEGER)
                    or isinstance(table.columns[column_name].type, FLOAT)
                ) and len(distinct_values) > num_max_distinct:
                    min_value = connection.execute(
                        select(func.min(table.columns[column_name]))
                    ).scalar()
                    max_value = connection.execute(
                        select(func.max(table.columns[column_name]))
                    ).scalar()
                    # column_values_dict[(table_name, column_name)] = [min_value, max_value]
                    column_values_dict[(table_name, column_name)] = (
                        f"Numerical values from min {min_value} to max {max_value}"
                    )
                # If non-numeric and distinct values greater than num_max_distinct, fetch random num_max_distinct values
                elif len(distinct_values) > num_max_distinct:
                    column_values_dict[(table_name, column_name)] = sample(
                        distinct_values, num_max_distinct
                    )
                else:
                    column_values_dict[(table_name, column_name)] = distinct_values

        return str(column_values_dict)

    def get_schema_str(self):
        if self.schema_str is not None:
            return self.schema_str
        schema_str = ""
        inspector = inspect(self._engine)
        table_names = inspector.get_table_names()

        for table_name in table_names:
            schema_str += f"TableName: {str(table_name)}\n"
            schema_str += "FieldName | FieldType | IsUnique\n"
            columns = inspector.get_columns(table_name)

            unique_keys = [
                uk["column_names"]
                for uk in inspector.get_unique_constraints(table_name)
            ]
            unique_columns = [col for sublist in unique_keys for col in sublist]

            for column in columns:
                # Getting data type of column
                FieldName = column["name"]
                FieldType = str(column["type"])
                IsUnique = column["name"] in unique_columns

                schema_str += f"{FieldName} | {FieldType} | {IsUnique}\n"
            schema_str += "\n"
        self.schema_str = schema_str
        return schema_str

    @staticmethod
    def _get_tables_using_skimage(df, using_empty_row_col=False):
        if using_empty_row_col:
            binary_rep = np.ones_like(df)
            for i in range(len(df.index)):
                if df.iloc[i, :].isnull().all():
                    binary_rep[i, :] = 0
            for j in range(len(df.columns)):
                if df.iloc[:, j].isnull().all():
                    binary_rep[:, j] = 0
        else:
            binary_rep = np.array(df.notnull().astype("int"))

        l = label(binary_rep)
        tables = []
        for s in regionprops(l):
            table = df.iloc[s.bbox[0] : s.bbox[2], s.bbox[1] : s.bbox[3]]
            tables.append(table)

        return tables

    @staticmethod
    def _add_header_to_table(table, num_headers):
        result_table = deepcopy(table)
        column_headers = [
            tuple(table.iloc[i, j] for i in range(num_headers))
            for j in range(table.shape[1])
        ]

        if num_headers == 1:
            result_table.columns = list(table.iloc[0, :])
        elif num_headers >= 2:
            result_table.columns = pd.MultiIndex.from_tuples(column_headers)

        result_table = result_table.iloc[num_headers:, :]
        return result_table


# %%

if __name__ == "__main__":

    filepath = "./datasets/AdventureWorks.xlsx"

    # Load data from excel or csv file, sqlite_in_memory=True will store data in memory
    # db = StructuredDatabase.from_excel_or_csv(filepath, sqlite_in_memory=False)
    
    # Load data from sqlite database
    db = StructuredDatabase.from_uri("sqlite:///./data/SalesAssistant.db")

    # Get schema string
    schema_str = db.get_schema_str()

    # Get data from database
    result = db.run("SELECT * FROM SalesRep")
    
    # Get data from database as pandas dataframe
    df = db.df_from_sql_query("SELECT * FROM SalesRep")
    
    # Fetch distinct row values
    distinct_row_values = db.fetch_distinct_row_values("[('SalesRep', 'Name')]")