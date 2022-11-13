import streamlit as st
from streamlit import session_state as ss
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import extra_streamlit_components as stx
import  streamlit_toggle as tog
from streamlit_elements import elements, mui, html, dashboard, editor, lazy, sync
import pandas as pd
import json
import requests
import openai
import stock_api

# primary_color = "#c76363"
# background_color = "#dacdcd"
# secondary_background_color = "#d2ae9c"
primary_color = "#91b36b"
background_color = "#FFFFFF"
secondary_background_color = "#F0F2F6"

mockData = [
    {
        "name": "Bayer",
        "symbol": "BAYRY",
        "score": 0.1,
        "qa_pairs": [
            {
                "question": "What is the company doing to reach net-zero targets?",
                "answer": "We are evaluating our supply chain and will be working with our suppliers to reduce their emissions.",
                "confidence": 0.4,
            },
            {
                "question": "What is your company's sustainability score?",
                "answer": "We have a sustainability score of 0.8",
                "confidence": 0.1,
            },
            {
                "question": "What materials do you use in your planes?",
                "answer": "We use a lot of aluminium and carbon fibre.",
                "confidence": 0.8,
            },
            {
                "question": "What materials do you use in your planes?",
                "answer": "We use a lot of aluminium and carbon fibre.",
                "confidence": 0.5,
            },
        ],
    },
    {
        "name": "Boeing",
        "symbol": "BA",
        "score": 0.5,
        "qa_pairs": [
            {
                "question": "What is the company doing to reach net-zero targets?",
                "answer": "We are evaluating our supply chain and will be working with our suppliers to reduce their emissions.",
                "confidence": 0.1,
            },
            {
                "question": "What is your company's sustainability score?",
                "answer": "We have a sustainability score of 0.8",
                "confidence": 0.1,
            },
            {
                "question": "What materials do you use in your planes?",
                "answer": "We use a lot of aluminium and carbon fibre.",
                "confidence": 0.5,
            },
            {
                "question": "What materials do you use in your planes?",
                "answer": "We use a lot of aluminium and carbon fibre.",
                "confidence": 0.8,
            },
        ],
    },
    {
        "name": "Apple",
        "symbol": "AAPL",
        "score": 0.9,
        "qa_pairs": [
            {
                "question": "What is the company doing to reach net-zero targets?",
                "answer": "We don't care abvout the environment!@",
                "confidence": 0.9,
            },
            {
                "question": "What is your company's sustainability score?",
                "answer": "It's something, not sure.",
                "confidence": 1,
            },
            {
                "question": "What materials do you use in your planes?",
                "answer": "Jell-o.",
                "confidence": 0.9,
            },
            {
                "question": "What materials do you use in your planes?",
                "answer": "We use a lot of aluminium and carbon fibre.",
                "confidence": 0.8,
            },
        ],
    },
]

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

available_companies = ["Bayer", "Boeing", "Apple"]

### Streamlit app starts here

c1 = st.container()
c2 = st.container()
c3 = st.container()
c4 = st.container()

def get_symbol_from_company(company):
    for companyInfo in mockData:
        if companyInfo["name"] == company:
            return companyInfo["symbol"]

    return ""

def format_market_cap(market_cap):
    if market_cap < 1000:
        rounded = round(market_cap, 2)
        return "$" + str(rounded) + " M"
    elif market_cap < 1000000:
        rounded = round(market_cap / 1000, 2)
        return "$" + str(rounded) + " B"
    else:
        rounded = round(market_cap / 1000000, 2)
        return "$" + str(rounded) + " T"

def get_investment_profile(company):
    with st.expander(label="Investment Profile"):
        company_info = stock_api.get_company_info(symbol=get_symbol_from_company(company))
        # Write and format exchange, country, market capitalization, and industry
        st.write("Exchange: " + company_info["exchange"])
        st.write("Country: " + company_info["country"])
        st.write("Market Capitalization: " + format_market_cap(company_info["marketCapitalization"]))
        st.write("Industry: " + company_info["finnhubIndustry"])

def get_confidence_style(qa_pair, bg_color):
    if "confidence" in qa_pair:
        conf = qa_pair["confidence"]
    else:
        conf = 0.5
    color = "rgb({},{},{},{})".format(145, 179, 107, conf)
    return f'radial-gradient(farthest-side at 40% 50%, {color}, {bg_color})'

# Share to social media
def compose_share_text():
    params = st.experimental_get_query_params()
    if "companies" in params:
        # Format a returned statement like "Here's a sustainability comparison of Apple, Boeing, and Bayer"
        companies = params["companies"]
        if len(companies) == 1:
            return "Here's a sustainability evaluation of " + companies[0] + ":"
        elif len(companies) == 2:
            return "Here's a sustainability comparison of " + companies[0] + " and " + companies[1] + ":"
        else:
            return "Here's a sustainability comparison of " + ", ".join(companies[:-1]) + ", and " + companies[-1]
    else:
        return "Check out this website to see how sustainable your favourite companies are!"

def compose_curr_url():
    domain = "localhost:8501"
    queryParams = []
    if "companies" in ss:
        for c in ss.companies:
            queryParams.append(f'companies={c}')
    
    queryStr = ""
    if len(queryParams) > 0:
        queryStr = "?" + "&".join(queryParams)

    # using domain and query params (map of query params), compose the current url
    return "http://" + domain + queryStr

def get_share_text():
    return """
                <div style="display:flex;margin:0px">
                    <div style="margin-top:12px;margin-right:10px">
                        <a class="twitter-share-button"
                            href="https://twitter.com/intent/tweet?text={text}"
                            data-url="{url}"
                            data-size="large">
                        Tweet</a>
                        <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    </div>
                    <div style="margin-top:15px;margin-right:10px">
                        <a data-size="large" data-url="{url}"/>
                        <script src="https://platform.linkedin.com/in.js" type="text/javascript"> lang: en_US</script>
                        <script type="IN/Share" data-url="{url}"></script>
                    </div>
                </div>
            """.format(text=compose_share_text(), url=compose_curr_url())

def get_share_elements():
    components.html(get_share_text())

# Mock function for now, will be an api call.
def get_company_info(company):
    for companyInfo in mockData:
        if companyInfo["name"] == company:
            return companyInfo

    return "{}"

def handle_company_select():
    st.experimental_set_query_params(companies=ss.companies)

    for company in ss.companies:
        if company not in ss:
            ss[company] = get_company_info(company)

# Get company info based on query params
params = st.experimental_get_query_params()
if "companies" in params:
    param_companies = params["companies"]
    ss.companies = param_companies
    for company in param_companies:
        if company not in ss:
            ss[company] = get_company_info(company)

with st.sidebar:
        page = option_menu(
            menu_title=None,
            options=["Company Lookup", "Trust & Data", "About Us"],
            icons=["house", "clipboard-data", "person-circle"],
        )

if page == "Company Lookup":
    with c1:
        st.title("Company Lookup")
        get_share_elements()
    
    with c2:
        title_column_1, title_column_2 = st.columns([8, 1])
        with title_column_1:
            st.multiselect("", available_companies, key="companies", on_change=handle_company_select)
        with title_column_2:
            st.markdown('#')
            tog.st_toggle_switch(label="Compare", 
                        key="compare", 
                        default_value=False, 
                        label_after = False, 
                        inactive_color = '#D3D3D3', 
                        active_color=primary_color, 
                        track_color=primary_color,
                        )
        st.markdown('#')

    with c3:
        params = st.experimental_get_query_params()
        param_companies = params["companies"] if "companies" in params else []
        if len(param_companies) > 0:
            # comparison mode
            if ss.compare:
                with elements("dashboard"):
                    if len(param_companies) > 0:
                        if "layout" not in ss:
                            ss.layout = []
                        
                        for company in param_companies:
                            # check whether company is already in layout
                            exists = False
                            for l in ss.layout:
                                if l["i"] == company:
                                    exists = True
                                    break
                            
                            # if not, add it
                            if not exists:
                                if ss[company]:
                                    height = 1 + len(ss[company]["qa_pairs"])/2
                                else:
                                    height = 2
                                ss.layout.append(dashboard.Item(company, 0, 0, 5, height, allowOverlap=True))
                            
                        with dashboard.Grid(ss.layout):
                            for company in param_companies:  
                                company_info = ss[company]                      
                                with mui.Card(key=company, sx={"display": "flex", "flexDirection": "column"}, raised=True):
                                    mui.CardHeader(title=company, subheader=f'Disclosure Score: {company_info["score"]}', sx={"color": "white", "background-color": primary_color, "padding": "5px 15px 5px 15px", "borderBottom": 2, "borderColor": "divider"})
                                    with mui.CardContent(sx={"flex": 1, "minHeight": 0, "background-color": secondary_background_color}):
                                        with mui.List():
                                            for qa_pair in company_info["qa_pairs"]:
                                                with mui.ListItem(sx={"background-image": get_confidence_style(qa_pair, secondary_background_color)}):
                                                    mui.ListItemText(primary= f'Q: {qa_pair["question"]}', secondary= f'A: {qa_pair["answer"]}', sx={"padding": "0px 0px 0px 0px"})
                                        # with mui.Accordion():
                                        #     with mui.AccordionSummary(expandIcon=mui.icon.ExpandMore):
                                        #         mui.Typography("FAQs")
                                        #     with mui.AccordionDetails():
                                        #         with mui.List():
                                        #             for qa_pair in company_info["qa_pairs"]:
                                        #                 with mui.ListItem(alignItems="flex-start", sx={"padding": "0px 0px 0px 0px"}):
                                        #                     mui.ListItemText(primary= f'Q: {qa_pair["question"]}', secondary= f'A: {qa_pair["answer"]}', sx={"padding": "0px 0px 0px 0px"})

                                    # with mui.CardActions(sx={"color": "white", "padding": "5px 15px 5px 15px", "background-color": "#ff4b4b", "borderTop": 2, "borderColor": "divider"}):
                                    #     mui.Button("Learn More", size="small", sx={"color": "white"})

            # tabular mode
            else:
                if "prev_company" in ss and ss.prev_company in param_companies:
                    df = ss.prev_company
                else:
                    df = param_companies[0]

                tabs = st.tabs(param_companies)
                for i, tab in enumerate(tabs):
                    with tab:
                        curr_company = param_companies[i]
                        company_info = ss[curr_company]

                        col1, _col, col3 = st.columns([1, 3, 1])

                        col1.subheader(company_info["name"])
                        col3.metric(label="Disclosure Score", value=company_info["score"])
                        for qa_pair in company_info["qa_pairs"]:
                            qa_html = """
                            <div style="margin:10px;background-image:{}">
                                <div style="font-weight: bold">Q: {}</div>
                                <div>A: {}</div>
                            </div>
                            """.format(get_confidence_style(qa_pair, background_color), qa_pair["question"], qa_pair["answer"])
                            st.markdown(qa_html, unsafe_allow_html=True)
                        st.markdown('#')
                        
                        get_investment_profile(curr_company)

                # ss.curr_company = stx.tab_bar(
                #     data=(stx.TabBarItemData(id=company, title=company, description="") for company in param_companies),
                #     default=df,
                # )

                # if ss.curr_company in ss:
                #     col1, _col, col3 = st.columns([1, 3, 1])
                #     company_info = ss[ss.curr_company]

                #     col1.subheader(company_info["name"])
                #     col3.metric(label="Disclosure Score", value=company_info["score"])
                #     for qa_pair in company_info["qa_pairs"]:
                #         qa_html = """
                #         <div style="margin:10px;background-image:{}">
                #             <div style="font-weight: bold">Q: {}</div>
                #             <div>A: {}</div>
                #         </div>
                #         """.format(get_confidence_style(qa_pair, background_color), qa_pair["question"], qa_pair["answer"])
                #         st.markdown(qa_html, unsafe_allow_html=True)
                #     st.markdown('#')
                    
                #     ss.prev_company = ss.curr_company

                #     get_investment_profile(ss.curr_company)
                # else:
                #     st.write("N/A")

    with c4:
        for i in range(6):
            st.markdown("#")
        



if page == "Trust & Data":
    st.title("Trust & Data")
    st.markdown("""
    Our data was taken from the following sources:
    - Company 10-K Filings
    - Company Sustainability Reports
    - Public Earnings Call Transcripts
    """)


if page == "About Us":
    st.title("About Us")
    st.write("This app was created by the Ungreenwash team.")
