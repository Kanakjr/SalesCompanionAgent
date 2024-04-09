examples = [
    {
        "input": "How am I performing against his goal",
        "description": "When Sales Rep want to know about his performance",
        "query": "WITH TargetCalls AS (\n    SELECT \n\t\tsr.Name,\n        sr.SalesRep_ID, \n        h.Priority,\n        SUM(pl.Calls) AS Target_Calls\n    FROM \n        main.SalesRep sr\n    JOIN \n        main.HCP h ON sr.SalesRep_ID = h.Territory_ID\n    JOIN \n        main.PriorityLookup pl ON h.Priority = pl.Priority\n\tWHERE\n\t\tsr.Name = 'Ethan Miller'\n    GROUP BY \n        sr.Name,sr.SalesRep_ID, sr.Territory_ID, h.Priority\n), TotalCallsMade AS (\n    SELECT \n        ih.SalesRep_ID, \n\t\th.Priority,\n        COUNT(ih.History_ID) AS Total_Calls_Made\n    FROM \n        main.InteractionHistory ih\n\t\tJOIN \n        main.HCP h ON h.HCP_ID = ih.HCP_ID\n\t\tJOIN\n\t\tmain.SalesRep sr ON sr.SalesRep_ID = ih.SalesRep_ID\n\tWHERE\n\t\tsr.Name = 'Ethan Miller'\n    GROUP BY \n        ih.SalesRep_ID, h.Priority\n)\nSELECT \n    tc.Name,\n\ttc.SalesRep_ID,\n    tc.Priority,\n\ttc.Target_Calls,\n    COALESCE(tcm.Total_Calls_Made, 0) AS Total_Calls_Made,\n    COALESCE((COALESCE(tcm.Total_Calls_Made, 0) * 100.0) / NULLIF(tc.Target_Calls, 0), 0) AS Percent_Achieved\nFROM \n    TargetCalls tc\nLEFT JOIN \n    TotalCallsMade tcm ON tc.SalesRep_ID = tcm.SalesRep_ID and tc.Priority = tcm.Priority\n\torder by tc.SalesRep_ID,tc.Priority;",
        "format_instruction": "**_{{Name}} Performance Report:_**  \n- **{{Priority}}:** Made **{{Total_Calls_Made}}** calls out of **{{Target_Calls}}** calls target and achieved **{{Percent_Achieved}}**.  \n"
    },
    {
        "input": "What I should know about Doctor [HCP Name]",
        "description": "When Sales Rep wants to know about doctors",
        "query": "Select HCP_Name,Speciality,Phone_No,Email,Account_Type,Account_Name from [main].[HCP] Where HCP_Name = 'William Davis';",
        "format_instruction": "{{HCP_Name}} is a {{Speciality}} specialist at {{Account_Name}}. You can contact him via phone at {{Phone_No}} or email at {{Email}}.\n  - **HCP Name**: {{HCP_Name}}\n  - **Specialty**: {{Speciality}}\n  - **Phone Number**: {{Phone_No}}\n  - **Email**: {{Email}}\n  - **Account Type**: {{Account_Type}}\n  - **Account Name**: {{Account_Name}}"
    },
    {
        "input": "Can you provide speaking notes for my Phone call/ Meeting with [HCP Name]?",
        "description": "Sales Rep is asking to prepare talking points to his meet with doctor/hcp",
        "query": "SELECT A.hcp_id, A.hcp_name,B.salesrep_id,B.name as SaleRep_Name\nFROM\n(SELECT DISTINCT hcp.hcp_id, hcp.hcp_name,hcp.Email\nFROM main.HCP hcp WHERE hcp.hcp_name = 'Mia Brown') A cross join \n(SELECT DISTINCT sr.salesrep_id, sr.name\nFROM main.SalesRep sr WHERE  sr.name = 'Bob Smith') B;",
        "format_instruction": ""
    },
    {
        "input": "Who should [SalesRep] contact this week?",
        "description": "Sales Rep is asking for meeting plan to whom he should meet",
        "query": "WITH LastInteraction AS (\n    SELECT ih.HCP_ID, count(*) Number_of_Interaction,MAX(ih.Contact_Date) AS Last_Interaction_Date\n    FROM \n        main.InteractionHistory as ih\n    GROUP BY ih.HCP_ID\n)\nSELECT \n    sr.Name,\n    h.HCP_ID,\n\th.HCP_Name,\n    h.Priority,\n    pl.Calls Target,\n\tCOALESCE(li.Number_of_Interaction,0) Number_of_Interaction,\n\tCOALESCE(li.Number_of_Interaction,0)*100 / pl.Calls AS '%Achieved',\n    DATEDIFF(DAY, COALESCE(li.Last_Interaction_Date,DATEADD(QUARTER,DATEDIFF(QUARTER,0,GETDATE()),0)), GETDATE()) AS Days_Since_Last_Interaction,\n    pl.Calls * DATEDIFF(DAY, COALESCE(li.Last_Interaction_Date,DATEADD(QUARTER,DATEDIFF(QUARTER,0,GETDATE()),0)) ,GETDATE()) AS Combined_Score\nFROM \n\tmain.SalesRep sr \n\tLEFT JOIN main.HCP h  ON sr.SalesRep_ID = h.Territory_ID\n\tLEFT JOIN LastInteraction li  ON li.HCP_ID = h.HCP_ID\n\tJOIN main.PriorityLookup pl on h.Priority = pl.Priority\nWHERE \n\tsr.Name = 'Ethan Miller' \nORDER BY \n    Combined_Score DESC;",
        "format_instruction": "Here is plan for [SalesRep]:  \n        1. Schedule a meeting with **{{HCP_Name}}** a priority **{{Priority}}** contact. It has been **{{Days_Since_Last_Interaction}}** since your last interaction. Your goal is to meet **{{Target}}** this quarter, and you have currently completed **{{Number_of_Interaction}}** of these meetings.  \n        2. Schedule a meeting with **{{HCP_Name}}** a priority **{{Priority}}** contact. It has been **{{Days_Since_Last_Interaction}}** since your last interaction. Your goal is to meet **{{Target}}** this quarter, and you have currently completed **{{Number_of_Interaction}}** of these meetings.  \n        "
    },
    {
        "input": "Can you Draft an email to [HCP Name]",
        "description": "Sales Rep is asking to draft or write an email",
        "query": "Select HCP_Name,Speciality,Phone_No,Email,Account_Type,Account_Name from [main].[HCP] Where HCP_Name = 'William Davis';",
        # "query": "SELECT A.hcp_id, A.hcp_name,B.salesrep_id,B.name as SaleRep_Name\nFROM\n(SELECT DISTINCT hcp.hcp_id, hcp.hcp_name,hcp.Email\nFROM main.HCP hcp WHERE hcp.hcp_name = 'Mia Brown') A cross join \n(SELECT DISTINCT sr.salesrep_id, sr.name\nFROM main.SalesRep sr WHERE  sr.name = 'Bob Smith') B;",
        "format_instruction": ""
    },
    {
        "input": "How many prescription has been written by [HCP Name]?",
        "description": "When Sales Rep want to know the prescription from given doctor",
        "query": "Select hcp.HCP_Name, SUM(P.TRx) Total_Number_Of_Prescription, SUM(P.NRx) Number_Of_New_Prescription from [main].[Prescription] P join main.HCP hcp on P.[HCP ID] = hcp.HCP_ID join [main].[SalesRep] sr on sr.SalesRep_ID = P.Territory_ID where hcp.HCP_Name = 'Crystal Esparza' Group by hcp.HCP_Name;",
        "format_instruction": ""
    },
    {
        "input": "Which doctors are assigned to me and what are their priority?",
        "description": "When Sales Rep want to know the doctors assigned to them",
        "query": "Select hcp.HCP_Name, Priority\nfrom main.HCP hcp join [main].[SalesRep] sr on hcp.Territory_ID = sr.Territory_ID\nwhere sr.Name = 'Dinesh Pal';",
        "format_instruction": ""
    }
]
