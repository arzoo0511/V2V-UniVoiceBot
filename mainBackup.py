import gradio as gr
import whisper
from tempfile import NamedTemporaryFile
import os
import asyncio
import edge_tts
from langsmith import traceable
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "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")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "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")
ffmpeg_path = r"C:\Users\ICG0148\Downloads\ffmpeg-7.1.1-full_build\ffmpeg-7.1.1-full_build\bin"
if ffmpeg_path not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + ffmpeg_path
llm = ChatGroq(api_key=os.environ["GROQ_API_KEY"], model="llama3-8b-8192")
whisper_model = whisper.load_model("large-v3")


custom_table_info = {"company_master": """
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
db = SQLDatabase.from_uri(
   ("postgresql+psycopg2://postgres:Anarza%40123@localhost:5432/financialdata"),
    include_tables=["company_master", "company_additional_details", "quarterly_results"],
    sample_rows_in_table_info=0,
    custom_table_info=custom_table_info
)
system_message = """
You are a world-class PostgreSQL expert data analyst and a friendly financial assistant.
You will answer user questions about financial data by running SQL queries against a database.
You have access to a set of tools to interact with the database.

Based on the user's question, decide whether it is necessary to use a tool.
If a tool is needed, you must use one.
If no tool is needed, you can answer directly.

When generating a SQL query, follow these rules:
1.  If the user gives an approximate company name (like "TCS" or "Enforcers"), you MUST first resolve the exact name using a query like: `SELECT comp_name FROM company_master WHERE comp_name ILIKE '%<input_name>%' LIMIT 1;`. Then use that exact result in subsequent queries.
2.  If asked a vague question like "Should I invest in TCS?", interpret it as a request for key financial indicators (net_sales, profit_after_tax, ebitda, trends) from the latest quarter. Do not give financial advice; just present the data.
3.  Handle common voice transcription errors like "TDL/TDC" for "TCS" and "Enforcers" for "Infosys".
4.  After executing the SQL, formulate a clear, user-friendly answer. DO NOT show the SQL query to the user. Just provide the final answer. If you cannot find data, say so politely.
5.  If the user asks about a company that does not exist in the database, politely inform them that you cannot find any information on that company.
6.  If the user asks for financial data for a specific year, ensure you filter results by that year.
7.  If the user asks for a specific financial metric (like EBITDA), ensure you retrieve it for the latest available date or the specified period.
8.  If the user asks for a comparison (like profit after tax for two years), ensure you retrieve and format the data clearly.
9.  If the user asks for a list of companies, ensure you retrieve the top results based on the specified criteria (like EBITDA or revenue).
10. If the user asks for a specific company's symbol, ensure you retrieve it from the company_master table.
11. If the user asks for a company's last traded price, ensure you retrieve it from the company_master table.
12. If the user asks for a company's financials for a specific year, ensure you filter results by that year and retrieve all relevant metrics.
13. If the user asks for a company's financials for a specific quarter, ensure you filter results by that quarter and retrieve all relevant metrics.
14. If the user asks for a company's debt to equity ratio, ensure you retrieve it from the latest quarterly results.
15. If the user asks for a company's EBITDA, ensure you retrieve it for the latest available date or the specified period.
16. If the user asks for a company's financials for a specific reporting period, ensure you filter results by that period and retrieve all relevant metrics.
17. If the user asks for a company's financials for a specific month, ensure you filter results by that month and retrieve all relevant metrics.
18. If the user asks for a company's financials for a specific quarter, ensure you filter results by that quarter and retrieve all relevant metrics.
19. If the user asks for a company's financials for a specific year, ensure you filter results by that year and retrieve all relevant metrics.
20. If the user asks for a company's financials for a specific month, ensure you filter results by that month and retrieve all relevant metrics.
21. If the user asks for a company's financials for a specific quarter, ensure you filter results by that quarter and retrieve all relevant metrics.
22. If the user asks for a company's financials for a specific year, ensure you filter results by that year and retrieve all relevant metrics.
23. If the user asks for a company's financials for a specific month, ensure you filter results by that month and retrieve all relevant metrics.
24. If the user asks for a company's financials for a specific quarter, ensure you filter results by that quarter and retrieve all relevant metrics.
25. If the user asks for a company's financials for a specific year, ensure you filter results by that year and retrieve all relevant metrics.


You have access to the following tables:
- company_master: Contains master data for publicly traded companies.
- company_additional_details: Contains dynamic, market-related and fundamental data updates.
- quarterly_results: Contains quarterly financial results.
You will answer questions by generating SQL queries to retrieve data from these tables.

You must use the following format for your responses:
Response: <your answer based on the SQL query results>

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
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"), 
    ]
)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    prompt=prompt,  
    verbose=True,
    agent_type="openai-tools",
    handle_parsing_errors=True
)

def correct_transcription(text):
    corrections = {
        "enforcers": "Infosys", "enforcer": "Infosys","in-courses": "Infosys", "in courses": "Infosys","infusions": "Infosys", "infusion": "Infosys",
        "tdl": "TCS", "tdc": "TCS", "t d l": "TCS", "t d c": "TCS",
        "natural data":"financial data", "natural data": "financial data", "natural data": "financial data",
        "pharmacies": "pharmaceuticals", "pharmacy": "pharmaceuticals",
    }
    lower_text = text.lower()
    for wrong, right in corrections.items():
        if wrong in lower_text:
            lower_text = lower_text.replace(wrong, right)
    return lower_text.capitalize()

def speech_to_text(audio_path):
    if not audio_path:
        return ""
    print(f"Transcribing audio from: {audio_path}")
    raw_text = whisper_model.transcribe(audio_path)["text"]
    corrected_text = correct_transcription(raw_text)
    print(f"Transcription: '{raw_text}' -> Corrected: '{corrected_text}'")
    return corrected_text

async def _text_to_speech_async(text, output_path):
    voice = "en-IN-NeerjaNeural"
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def text_to_speech(text):
    if not text:
        return None
    output_file = NamedTemporaryFile(suffix=".mp3", delete=False)
    output_path = output_file.name
    output_file.close()

    print(f"Generating speech for: '{text}'")
    try:
        asyncio.run(_text_to_speech_async(text, output_path))
        print(f"Speech saved to: {output_path}")
        return output_path
    except Exception as e:
        return None

@traceable(name="financial_chatbot_pipeline")
def chatbot_pipeline(audio_path):
    try:
        transcription = speech_to_text(audio_path)
        if not transcription.strip():
            return "Could not understand audio.", "I didn't catch that. Could you please speak clearly?", None
        result = agent_executor.invoke({"input": transcription})
        response_text = result["output"]
        print(f"Agent Response: {response_text}")
        response_audio = text_to_speech(response_text)
        return transcription, response_text, response_audio
    except Exception as e:
        error_message = f"An error occurred in the pipeline: {str(e)}"
        print(error_message)
        user_error_text = "Sorry, I ran into a technical problem. Please try asking your question again."
        error_audio = text_to_speech(user_error_text)
        return "Error", user_error_text, error_audio
import gradio.networking
def get_dummy_ip():
    return "127.0.0.1"
gradio.networking.get_local_ip_address = get_dummy_ip

with gr.Blocks(theme=gr.themes.Soft(), css="footer {visibility: hidden}") as demo:
    gr.Markdown("""
    <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
        <h1 style="font-size: 2.5em; color: #343a40;">Real-Time V2V UnivestBot</h1>
        <p style="font-size: 1.2em; color: #6c757d;">Ask anything about Indian financial data using your voice.</p>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Speak Your Financial Query")
            submit_btn = gr.Button("Get Answer", variant="primary")
            status = gr.Textbox(label="Status", value="Ready for your query...", interactive=False)

        with gr.Column(scale=2):
            transcript = gr.Textbox(label="You Said", interactive=False)
            bot_response = gr.Textbox(label="Bot's Reply", interactive=False, lines=5)
            audio_output = gr.Audio(label="Bot Speaking", type="filepath", autoplay=True)

    def update_ui(audio_filepath):
        if not audio_filepath:
            return "No audio recorded.", "Please record your voice first.", None, "Error: No audio input."
        status.value = "Processing your request..."
        transcript_val, reply_val, audio_val = chatbot_pipeline(audio_filepath)
        status.value = "Done!"
        return transcript_val, reply_val, audio_val, "Done!"

    submit_btn.click(
        fn=update_ui,
        inputs=[audio_input],
        outputs=[transcript, bot_response, audio_output, status]
    )
demo.launch(share=False, server_name="127.0.0.1", server_port=7860)