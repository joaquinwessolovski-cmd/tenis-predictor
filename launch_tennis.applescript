tell application "Terminal"
    do script "cd /Users/wesso/Downloads/tenis && source venv/bin/activate && streamlit run streamlit_app.py"
    activate
end tell
