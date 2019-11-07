mkdir -p ~/.streamlit/

spacy download en_core_web_lg

echo "\
[general]\n\
email = \"email@website.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml