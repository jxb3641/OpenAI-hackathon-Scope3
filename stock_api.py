import finnhub
import streamlit as st

api_key = "cdo2dj2ad3i5o5oktnn0cdo2dj2ad3i5o5oktnng"
finnhub_client = finnhub.Client(api_key=api_key)

def get_symbol(company):
    """Get the symbol of a company.

    Args:
        company (str): company name.

    Returns:
        str: company symbol.
    """

    # get company symbol
    symbol = finnhub_client.symbol_lookup(company)
    return symbol

def get_company_info(symbol):
    """Get company info.

    Args:
        company (str): company name.

    Returns:
        dict: company info.
    """

    # get company profile
    profile = finnhub_client.company_profile2(symbol=symbol)
    return profile

def get_peers(symbol):
    """Get company peers.

    Args:
        company (str): company name.

    Returns:
        list: company peers.
    """

    # get company peers
    peers = finnhub_client.company_peers(symbol)
    return peers