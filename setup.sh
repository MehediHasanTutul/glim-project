mkdir -p ~/.streamlit/

echo "[theme]
primaryColor="#59708C"
backgroundColor="#EDDCF1"
secondaryBackgroundColor="#F1E5F1"
textColor="#051614"
font="sans serif"
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml
