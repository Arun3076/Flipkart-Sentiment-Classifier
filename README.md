🛒 Flipkart Sentiment Classifier

This project is a Transformer-based sentiment classifier for analyzing Flipkart product reviews. It predicts whether a review is positive, negative, or neutral.
Unlike the Amazon and Airline projects, this one comes with a Streamlit interface for easy interaction.

📂 Project Structure:

Flipkart-Sentiment-Classifier/

│── app.py                          # Streamlit app for sentiment prediction

│── Flipcart_Model.ipynb            # Jupyter Notebook (training + testing)

│── flipkart_sentiment_balanced.csv # Flipkart reviews dataset

│── fp-model/                       # (Not uploaded, too large for GitHub)

│── README.md                       # Documentation


⚡ Features

Built using Hugging Face Transformers.

Trained on a balanced Flipkart reviews dataset.

Provides an interactive web app powered by Streamlit.

Jupyter Notebook (Flipcart_Model.ipynb) contains training and evaluation pipeline.

🚀 How to Run

Since the fine-tuned model is very large (~255 MB), it is not uploaded to GitHub.
The notebook and Streamlit app will automatically download the pre-trained model from Hugging Face.

1️⃣ Clone the repository

git clone https://github.com/your-username/Flipkart-Sentiment-Classifier.git
cd Flipkart-Sentiment-Classifier

2️⃣ Install dependencies

pip install -r requirements.txt

3️⃣ Run the Jupyter Notebook

jupyter notebook Flipcart_Model.ipynb

Executes training and testing pipeline.

Hugging Face will automatically download the base model.

4️⃣ Run the Streamlit App
streamlit run app.py

This will open a local web interface in your browser, where you can input any Flipkart review and get sentiment predictions.

📌 Notes

The fp-model folder is not included due to GitHub file size limits.

Hugging Face downloads the required Transformer model automatically on first run.

To share your fine-tuned model, you can upload it to Hugging Face Hub or Google Drive, then update app.py to load it from there.

1️⃣ Clone the repository
