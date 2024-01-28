import evaluate
import datetime
import torch
import transformers
import mlflow
import mlflow.pytorch
import os
import re
import streamlit as st


# Increse timeout to download artifacts from MLFlow server
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "600"

class MLFlow_server():
    def __init__(self,
                 host:str="mlflow",
                 port:str="5000",
                 experiment:str="sdg_experiment"):
        
        self.server_ip = "http://" + ':'.join([host, port])
        self.experiment_name = experiment

        # Initialize best model and best run
        self.best_model = self.best_run = ""

        # Initialize tokenizer and base model
        self.tokenizer = self.base_model = ""

        try:
            # Connect to MLFlow server
            mlflow.set_tracking_uri(self.server_ip)
            mlflow.set_experiment(self.experiment_name)
        except:
            print(f"[ERROR] Connection to {self.server_ip} cannot be established!")
    


    def get_best_run(self):
        # Get best run
        experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
        runs = mlflow.search_runs(experiment_ids=experiment_id)
        
        # Use F1 metric to determine best run
        self.best_run = runs.loc[runs['metrics.f1'].idxmax()]

        # Get best model
        self.best_model = f"runs:/{self.best_run['run_id']}/model"
        print(f"[INFO] Loaded best model with F1 score: {round(self.best_run['metrics.f1'], 5)}")

        # Load model and tokenizer
        self.load_model_tokenizer()


    def load_model_tokenizer(self):
        # Load device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.base_model = self.best_run["params.base_model"]
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.base_model)

        # Load model
        self.loaded_model = mlflow.pytorch.load_model(self.best_model)
        self.loaded_model.to(self.device)


    def clear_text(self, text):
        # Remove commas
        clean_text = re.sub(r',', '', text)
        # Remove full stops
        clean_text = re.sub(r'\.', '', clean_text)
        # Remove single quotes and double quotes
        clean_text = re.sub(r"['\"]", '', clean_text)
        # Remove other non-word characters
        clean_text = re.sub(r'\W', ' ', clean_text)
        # Remove extra spaces
        clean_text = re.sub(r'\s+', ' ', clean_text)

        # Tokenize text
        tokenized_input = self.tokenizer(text, return_tensors="pt").to(self.device)

        return tokenized_input, clean_text


    def inference(self, input_text:str):
        # Tokenize text
        tokenized_input, clean_text = self.clear_text(input_text)

        with torch.no_grad():
            logits = self.loaded_model(**tokenized_input).logits
        
        # Get probabilities of all categories
        prob = torch.softmax(logits, dim=1)
        
        return prob, clean_text



class StreamlitApp():
    def __init__(self):
        st.set_page_config(
            layout="centered", 
            page_title="SDG Client Page"
        )
        st.caption("")
        st.title("News Classifier")

        # App interactions don't reset app
        if not "valid_inputs_received" in st.session_state:
            st.session_state["valid_inputs_received"] = False

        # Subtitle text
        st.write("")
        st.markdown("""Classify news using Titles and Text""")
        st.write("")

    def check_model_updates(self):
        try:
            # Connect to MLFlow server
            self.mlflow_server = MLFlow_server()
            
            # Get best News Classifier run in MLFlow server
            self.mlflow_server.get_best_run()
        except:
            print("[ERROR] Cannot load best model!")

    def get_current_time(self):
        return str(datetime.datetime.now())

    def funct_button_updates(self):
        # Time when update search is submitted
        current_time = self.get_current_time()

        # Check best model in MLFlow server
        self.check_model_updates()

        st.success(f"Checked at {current_time}. Model loaded: {self.mlflow_server.best_model}")

        st.session_state.valid_inputs_received = True
    
    def show_form(self):

        if st.button("Check model updates"):
            self.funct_button_updates()

        with st.form(key="my_form"):
            # Stores final results
            results = {}
            clean_text = ""

            # Pre-defined text that can be used as example
            pre_defined_text = "Even with service restored , it will take about 2 weeks for gasoline \
                in Houston to reach East Coast filling @ @"

            # Text box to input title + news text
            text = st.text_area(
                # Instructions
                "Enter title and news as a single text string",
                # 'sample' variable that contains our keyphrases.
                pre_defined_text,
                # The height
                height=200,
                # The tooltip displayed when the user hovers over the text area.
                help="You can only process one news item at a time. The text will be considered as title + news",
                key="1",
            )

            # Create submit button
            submit_button = st.form_submit_button(label="Submit")

            # If the user has pressed "submit" without text, the app will display a warning message
            if not submit_button and not st.session_state.valid_inputs_received:
                st.stop()

            elif submit_button and not text:
                st.warning("There is no text to classify")
                st.session_state.valid_inputs_received = False
                st.stop()

            elif submit_button or st.session_state.valid_inputs_received:

                if submit_button:
                    
                    if not hasattr(self, "mlflow_server"):
                        self.funct_button_updates()
                    
                    # Store user's valid input 
                    st.session_state.valid_inputs_received = True

                    # Predict title + news
                    prob_tensor, clean_text = self.mlflow_server.inference(text)
                    
                    # Category probability as float
                    prob_float = list(prob_tensor.cpu().numpy()[0].tolist())
                    
                    # Category probability as percentage
                    prob = [f"{x:.2%}" for x in prob_float]
                    
                    # Add categories to percentage
                    labels = ["Category " + x for x in list(self.mlflow_server.loaded_model.config.label2id.keys())]
                    results = dict(zip(labels, prob))

                    # Show text before classify
                    with st.expander("Clean text before tokenize"):
                        st.write(clean_text)

                    # Print success message and results
                    st.success("Done!")

                    st.caption("")
                    st.markdown("### Check the results!")
                    st.caption("")

                    st.write(results)


if __name__ == "__main__":
    # Create Streamlit App
    app = StreamlitApp()

    # Show form to input text
    app.show_form()

