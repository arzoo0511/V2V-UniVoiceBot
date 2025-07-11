import gradio as gr
import whisper
from tempfile import NamedTemporaryFile
import os
from langsmith import traceable
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import pyttsx3
import asyncio
import edge_tts
import os

os.environ["PATH"] += os.pathsep + r"C:\\Users\\ICG0148\\Downloads\\ffmpeg-7.1.1-full_build\\ffmpeg-7.1.1-full_build\\bin"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["GROQ_API_KEY"] = "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"
llm = ChatGroq(api_key=os.environ["GROQ_API_KEY"], model="llama3-8b-8192")
whisper_model = whisper.load_model("large-v3")

custom_table_info = {
    "company_master": """
        This table contains master data for publicly traded companies.
        IMPORTANT COLUMNS:
        - comp_name: The full, official name of the company exactly as stored in the database. DO NOT guess or shorten the name.
            If the user gives an approximate name (like \"TCS\" or \"Tata Consultancy Services\"), FIRST resolve the exact name using: SELECT comp_name  FROM company_master WHERE comp_name ILIKE '%<input_name>%' LIMIT 1;
            Then use the **exact result of that query** in the final SQL.
        - symbol: The stock trading ticker symbol on the NSE.
        - industry: The general industry (e.g., 'Banks', 'Software').
        - sector: A more specific business sector (e.g., 'IT - Software').
        - bse_ltp_price: The last traded price on BSE.
    """,

    "company_additional_details": """
        This table contains dynamic, market-related and fundamental data updates.
        IMPORTANT COLUMNS:
        -nifty_500 : represents the float-weighted average of 500 of the largest Indian companies listed on the National Stock Exchange in india.
        -nifty_50 : represents the float-weighted average of 50 of the largest Indian companies listed on the National Stock Exchange in india.
        - fin_code: Identifier linking to the master table.
        - nse_ltp_price / bse_ltp_price: Last traded prices.
        - fundamental_score: Company rating.
        -short_term_trend, long_term_trend: changes in the company over a short period of time and over a long period of time .
        - verdicts: Investment sentiments.
        - founded_year, managing_director, incorporation_date: Company metadata.
    """,

    "quarterly_results": """
        This table contains quarterly financial results.
        IMPORTANT COLUMNS:
        - fin_code: Company identifier.
        - date_end: Format YYYYMM (202403 = March 2024).
        - type: Filing type ('C' for Consolidated).
        - no_of_months: Duration covered (3, 6, 12).
        - ebitda, net_sales, total_income, profit_after_tax, debt_equity_ratio , gross_npa,operating_profit_percent: Financial metrics/data/information.
    """
}
sql_prompt = ChatPromptTemplate.from_template("""
You are a world-class PostgreSQL expert data analyst. Convert the user's question into a valid SQL query.
Only respond with SQL. GENERATE PROPER QUERIES AFTER UNDERSTANDING THE DATABASE PROPERLY.                                                
Use these tables: `company_master`, `company_additional_details`, `quarterly_results`.
USE COLUMNS PRESENT IN THE DATABASE ONLY.         
IF "TDL"OR"TDC" IN SPEECH THEN TAKE IT AS "TCS" !!!
IF "ENFORCERS" IN SPEECH THEN TAKE IT AS "INFOSYS" !!!
⚠️ DO NOT return any extra text like explanations, comments, or "Note: ...". Only return the SQL query itself. Not even a semicolon is required at the end.

You are also a financial assistant that answers user queries by generating SQL from natural language, running it, and returning user-friendly results.

The user will ask questions about companies (e.g., TCS, Infosys, Reliance).

For each question:
1. Identify the intent (e.g., investment advice, stock price, financials).
2. If the user asks something vague like “Should I invest in TCS?”, interpret that as a request to retrieve the latest available financial indicators to help the user make an informed decision. DO NOT say you don’t understand.
3. Retrieve the following financial data if applicable:
   - net_sales
   - nifty_50
   - nifty_500                                                                                                         
   - total_income
   - Profit After Tax (profit_after_tax)
   - operating_profit_margin
   - ebitda                                                                                              
   - Short-term and Long-term Trend
4. Use the correct tables:
   - First check `company_master` and `quarterly_results` for financials.
   - Then check `company_additional_details` for trend and price-related info.
5. Always use the **latest** available data (MAX(date_end)) if no year is mentioned.
6. Always use the **official company name** from the `company_master` table (e.g., 'Tata Consultancy Services Ltd.').

Only return **ONE** query at a time. Use subqueries to get `fin_code`. No text output or commentary is allowed.

---

Examples:
Question: What is the EBITDA of Tata Consultancy Services Ltd. over the past 5 years?
SQL Query:
SELECT qr.date_end,qr.ebitda
FROM
  company_master cm
JOIN
  quarterly_results qr ON cm.fin_code = qr.fin_code
WHERE
  cm.comp_name = 'Tata Consultancy Services Ltd.' AND qr.type = 'C' AND CAST(LEFT(qr.date_end::text, 4) AS INT) >= EXTRACT(YEAR FROM CURRENT_DATE) - 5
ORDER BY
  qr.date_end DESC
                                                       
Question: Top 3 companies on the nifty 50 index?
SQL Query:
SELECT cm.comp_name, qr.total_income FROM company_master cm
JOIN quarterly_results qr ON cm.fin_code = qr.fin_code
JOIN company_additional_details cad ON cm.fin_code = cad.fin_code
WHERE cad.nifty_50 = 'Y' AND qr.type = 'C' AND qr.date_end = (SELECT MAX(date_end) FROM quarterly_results WHERE type = 'C') ORDER BY qr.total_income DESC LIMIT 3;

Question: Should I invest in Oil & Natural Gas Corporation Ltd.?
SQL Query:
SELECT 
  qr.net_sales,cad.nifty_50,cad.nifty_500,qr.total_income,qr.profit_after_tax,qr.ebitda,cad.short_term_trend,cad.long_term_trend
FROM
  company_master cm
JOIN
  quarterly_results qr ON cm.fin_code = qr.fin_code
JOIN
  company_additional_details cad ON cm.fin_code = cad.fin_code
WHERE
  cm.comp_name = 'Oil & Natural Gas Corporation Ltd.'
  AND qr.type = 'C'
  AND qr.date_end = (SELECT MAX(date_end)FROM quarterly_results WHERE fin_code = cm.fin_code AND type = 'C')
                                                                                                                                                                                                                       

Question: What is the last traded price for Tata Consultancy Services?  
SQL Query:  
SELECT bse_ltp_price FROM company_master WHERE comp_name = 'Tata Consultancy Services Ltd.'
                                                       
Question: Compare the profit after tax for HDFC Bank Ltd. in 2023 and 2024.  
SQL Query:  
SELECT '2023' AS year, profit_after_tax FROM quarterly_results 
WHERE fin_code = (SELECT fin_code FROM company_master WHERE comp_name = 'HDFC Bank Ltd.') 
AND type = 'C' AND date_end = (SELECT MAX(date_end) FROM quarterly_results 
WHERE fin_code = (SELECT fin_code FROM company_master WHERE comp_name = 'HDFC Bank Ltd.') 
AND type = 'C' AND CAST(date_end AS TEXT) LIKE '2023%') 
UNION ALL 
SELECT '2024' AS year, profit_after_tax FROM quarterly_results 
WHERE fin_code = (SELECT fin_code FROM company_master WHERE comp_name = 'HDFC Bank Ltd.') 
AND type = 'C' AND date_end = (SELECT MAX(date_end) FROM quarterly_results 
WHERE fin_code = (SELECT fin_code FROM company_master WHERE comp_name = 'HDFC Bank Ltd.') 
AND type = 'C' AND CAST(date_end AS TEXT) LIKE '2024%')

Question: Show the top 3 IT sector companies by EBITDA.  
SQL Query:  
SELECT cm.comp_name, qr.ebitda FROM company_master cm 
JOIN quarterly_results qr ON cm.fin_code = qr.fin_code 
WHERE cm.sector ILIKE '%IT%' AND qr.type = 'C' 
AND qr.date_end = (SELECT MAX(date_end) FROM quarterly_results WHERE type = 'C') 
ORDER BY qr.ebitda DESC LIMIT 3

Question: List 3 companies in the ‘IT - Software’ sector with revenue over ₹10,000 crore.  
SQL Query:  
SELECT cm.comp_name, qr.total_income 
FROM company_master cm 
JOIN quarterly_results qr ON cm.fin_code = qr.fin_code 
WHERE cm.sector = 'IT - Software' 
AND qr.type = 'C' 
AND qr.date_end = (SELECT MAX(date_end) FROM quarterly_results WHERE type = 'C') 
AND qr.total_income > 100000000000

Question: What is the symbol for Infosys?  
SQL Query:  
SELECT symbol FROM company_master WHERE comp_name = 'Infosys Ltd.'

Question: What was the profit after tax for Infosys in Q4 2023?  
SQL Query:  
SELECT profit_after_tax FROM quarterly_results WHERE fin_code = (SELECT fin_code FROM company_master WHERE comp_name = 'Infosys Ltd.') AND date_end = 202312

Question: What is the latest EBITDA of Infosys?  
SQL Query:  
SELECT ebitda, no_of_months FROM quarterly_results WHERE fin_code = (SELECT fin_code FROM company_master WHERE comp_name = 'Infosys Ltd.') AND type = 'C' AND date_end = (SELECT MAX(date_end) FROM quarterly_results WHERE fin_code = (SELECT fin_code FROM company_master WHERE comp_name = 'Infosys Ltd.') AND type = 'C') AND no_of_months = (SELECT MAX(no_of_months) FROM quarterly_results WHERE fin_code = (SELECT fin_code FROM company_master WHERE comp_name = 'Infosys Ltd.') AND type = 'C' AND date_end = (SELECT MAX(date_end) FROM quarterly_results WHERE fin_code = (SELECT fin_code FROM company_master WHERE comp_name = 'Infosys Ltd.') AND type = 'C'))

Question: What is the EBITDA of TCS for different reporting periods?  
SQL Query:  
SELECT ebitda, no_of_months FROM quarterly_results WHERE fin_code = (SELECT fin_code FROM company_master WHERE comp_name = 'Tata Consultancy Services Ltd.') AND type = 'C' AND date_end IN (SELECT MAX(date_end) FROM quarterly_results WHERE fin_code = (SELECT fin_code FROM company_master WHERE comp_name = 'Tata Consultancy Services Ltd.') AND type = 'C' AND no_of_months IN (3, 6, 12) GROUP BY no_of_months)

Question: ebitda of tcs 2025?
SQL Query:SELECT date_end, ebitda, no_of_months FROM quarterly_results WHERE fin_code = (SELECT fin_code FROM company_master WHERE comp_name = 'Tata Consultancy Services Ltd.')
AND type = 'C'AND CAST(date_end AS TEXT) LIKE '2025%'ORDER BY date_end;
                                                                                                        
Question: What is the debt to equity ratio of HDFC Bank in the latest consolidated quarter?  
SQL Query:  
SELECT debt_equity_ratio FROM quarterly_results WHERE fin_code = (SELECT fin_code FROM company_master WHERE comp_name = 'HDFC Bank Ltd.') AND type = 'C' AND date_end = (SELECT MAX(date_end) FROM quarterly_results WHERE fin_code = (SELECT fin_code FROM company_master WHERE comp_name = 'HDFC Bank Ltd.') AND type = 'C')

Question: Show me the 2025 financials of Tata Motors?
SQL Query:                      
SELECT 
  qr.net_sales,qr.total_income,qr.profit_after_tax,qr.ebitda,qr.operating_profit_percent,cad.nifty_50,cad.nifty_500,cad.short_term_trend,cad.long_term_trend
FROM
  company_master cm
JOIN
  quarterly_results qr ON cm.fin_code = qr.fin_code
JOIN
  company_additional_details cad ON cm.fin_code = cad.fin_code
WHERE
  cm.comp_name = 'Tata Motors Ltd.' AND qr.type = 'C' AND CAST(qr.date_end AS TEXT) LIKE '2025%'
                                              
---

Question: {question}  
SQL Query:
""")

rephrase_prompt_template = ChatPromptTemplate.from_template("""
You are a friendly financial assistant. Translate the user's question and SQL result into a helpful, clear answer.
Avoid mentioning SQL or tables.
If no result, say: "I couldn't find any information for that query. Please try rephrasing your question."

Original Question: {question}
Raw Database Result:
{result}

Helpful, User-Friendly Answer:
""")
db = SQLDatabase.from_uri(
   ("postgresql+psycopg2://postgres:Anarza%40123@localhost:5432/financialdata"),
    include_tables=["company_master", "company_additional_details", "quarterly_results"],
    sample_rows_in_table_info=2,
    custom_table_info=custom_table_info
)
query_chain = sql_prompt | llm | StrOutputParser()


def run_sql_and_summarize(inputs):
    question = inputs["question"].strip()
    try:
        sql = query_chain.invoke({"question": question})
        print(f"\n Generated SQL: {sql}\n")

        if not sql.strip().lower().startswith("select") or "from" not in sql.lower():
            return "Sorry, I couldn’t understand your question well enough to generate a valid SQL query. Please try rephrasing."

        result = db.run(sql)

    except Exception as e:
        return f"SQL Error: {str(e)}"

    try:
        summary_prompt = ChatPromptTemplate.from_template("""
        You are a helpful financial assistant. Here is a user question and SQL query result:
        Question: {question}
        SQL Result: {result}
        Give a short, natural language answer.
        """)
        summary_chain = summary_prompt | llm | StrOutputParser()
        return summary_chain.invoke({"question": question, "result": result})
    except Exception as e:
        return f"I got the SQL result but couldn't summarize it: {str(e)}"


full_chain = RunnableLambda(run_sql_and_summarize)


def correct_transcription(text):
    corrections = {
        "enforcers": "Infosys",
        "tdl": "TCS",
        "tdc": "TCS",
        "t d l": "TCS",
        "t d c": "TCS",
        "enforcer": "Infosys"
    }
    for wrong, right in corrections.items():
        text = text.lower().replace(wrong.lower(), right)
    return text


def speech_to_text(audio_path):
    raw_text = whisper_model.transcribe(audio_path)["text"]
    return correct_transcription(raw_text)


def generate_sql_response(text):
    return full_chain.invoke({"question": text})


def text_to_speech(text):
    output_file = NamedTemporaryFile(suffix=".wav", delete=False)
    output_path = output_file.name
    output_file.close()  

    try:
        engine = pyttsx3.init()
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        return output_path
    except Exception as e:
        print(f"TTS Error: {e}")
        return None


@traceable(name="chatbot_pipeline")
def chatbot_pipeline(audio_path):
    try:
        transcription = speech_to_text(audio_path)
        response = generate_sql_response(transcription)
        response_audio = text_to_speech(response)
        return transcription, response, response_audio
    except Exception as e:
        return str(e), "", None


import gradio.networking

def get_dummy_ip():
    return "127.0.0.1"
gradio.networking.get_local_ip_address = get_dummy_ip


with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("""
        <div style="text-align: center; padding: 10px">
            <h1 style="font-size: 2.5em;">Real-Time V2V UnivestBot</h1>
            <p style="font-size: 1.2em;">Ask anything about Indian financial data using your voice.</p>
        </div>
    """)

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Speak Your Financial Query", show_label=True)
            submit_btn = gr.Button("Get Answer")

        with gr.Column():
            transcript = gr.Textbox(label="You Said", interactive=False)
            bot_response = gr.Textbox(label="Bot's Reply", interactive=False)
            audio_output = gr.Audio(label="Bot Speaking", type="filepath")

    status = gr.Textbox(label="Status", value="Waiting for input...", interactive=False)

    def update_ui(audio):
        try:
            status_val = "Transcribing..."
            transcript_val = speech_to_text(audio)
            status_val = "Generating SQL + Answer..."
            reply_val = generate_sql_response(transcript_val)
            status_val = "Converting to speech..."
            audio_val = text_to_speech(reply_val)
            status_val = "Done"
            return transcript_val, reply_val, audio_val, status_val
        except Exception as e:
            return "Error", str(e), None, "Failed"

    submit_btn.click(
        fn=update_ui,
        inputs=[audio_input],
        outputs=[transcript, bot_response, audio_output, status]
    )


demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
