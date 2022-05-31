mkdir -p ~/.streamlit/

echo "[theme]
primaryColor='#59708C'
backgroundColor='#ECFCF3'
secondaryBackgroundColor='#ECFCF9'
textColor='#051614'
font='sans serif'
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml
