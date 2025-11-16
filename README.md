# spotify-suggestive-local


this tiny script analyzes your spotify extended streaming history(which you can request to download thru their website/services) using local tools and generates yearly insights, visual charts, summaries, recommendations, and playful roasts. It works fully offline using a llm through ollama. le script creates charts for overall listening patterns and detailed yearly breakdowns, then stores everything in a charts folder inside your history directory.

tools/libs used : matplotlib, pandas, glob, json, random, os, torch (optional), requests, Ollama, llama3 local model

an example of one of the outputs for say example: year 2025 -> a simple bar chart emphasizing listening trends, a local llm powered roast and some recs: <img width="3600" height="2000" alt="year_2025_summary" src="https://github.com/user-attachments/assets/0ed3d37c-811f-42d0-9fc5-2efde381ea8e" />


key inspiration behind this :

<img width="300" height="370" alt="Screenshot 2025-11-15 at 4 36 15 PM" src="https://github.com/user-attachments/assets/06c668de-bddb-4e71-b58f-ff3524d58400" />



one day....i will containerize this...
