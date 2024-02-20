from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
import argparse
import torch
from transformers import AutoTokenizer, pipeline
import transformers
from langchain_core.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from huggingface_hub import login, notebook_login
import csv
from tqdm import tqdm
import re
# token = "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"
# login(token)
from BatchwithLangChain import batches





class EmotionTransfer:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map='auto',
            max_length=500,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipeline, model_kwargs={'temperature': 0})

        self.first_template = """
            You are an affective computing expert who can rewrite a text passage in an emotional way.
            You will only reply with the transformed sentence in the emotional way as indicated from the emotion.
            You will try to add the correct emotional cues transforming the sentene in the targeted emotion.
            Re write the following text delimited by triple backquotes in a paraphrased {style}-ish way.


            ```{text}```


            Response:
        """


        self.second_template = """
        You are an affective computing expert, and psychologist who makes research on natural language processing and affective computing
        Your task is to inject accurate and affective the requested emotion into the given phrase.
        Convert the following phrase from neutral into an emotional very {style} textual style following the specific rules:
        1. Output only text.
        2. Only one phrase.
        3. Without emojis, faces, pictures or irrelevant characters.:

        'Neutral Phrase: {text}'
        'New Phrase:'
        """

        self.third_template = """
            You are an affective computing expert, and psychologist who makes research on natural language processing and affective computing.
            Your task is to inject hardcore and affective emotion into the sentences without using emojis, symbols, or any non-textual characters.
            Convert the following phrase from neutral into an emotional very {style} textual style following the specific rules:
            1. Output only text.
            2. Only one phrase.
            3. Do not use emojis, symbols, or any non-textual characters.
            4. Focus on using descriptive words and phrases to convey the emotion strongly.

            'Neutral Phrase: {text}'
            'New Phrase:'
        """
        self.fourth_template = """
            [INST]<<SYS>>As an expert in affective computing and psychology, your task is to transform neutral sentences into emotionally charged versions \
            without using emojis, symbols, or any forms of non-verbal expressions. \
            Focus solely on enhancing the text with emotional depth through the use of vivid, \
            descriptive language that conveys the specified emotion clearly and strongly. \n
            Please rewrite in just one short sentence the following neutral sentence.<</SYS>>
            [/INST] \
            Express a strong sense of "{style}" emotion. \Adapt the new phrase into a {language_style} style of language. \
            Do not explain or add anything else but the new sentence. \
            The output should be in plain text while a short sentence, relying entirely on language to communicate the emotion: \

            [NEUTRAL] "{text}"
            [/NEUTRAL]
            [NEW PHRASE]:

        """
        self.fifth_template = """
            [INST]<<SYS>>You are an expert in affective computing and text style tranfer expert, your task is to transform neutral sentences into emotionally charged versions \
            without using emojis, symbols, or any forms of non-verbal expressions. \

            Focus on enhancing the text with the use of {language_style}, \
            language that conveys the specified style of language clearly and strongly. \n
            Please rewrite in just one short sentence the following neutral sentence.<</SYS>>
            [/INST] \
            Express also a strong sense of "{style}" emotion. \
            Do not explain or add anything else but the new sentence. \
            The output should be in plain text while a short sentence, relying entirely on language to communicate the emotion: \

            [NEUTRAL] "{text}"
            [/NEUTRAL]
            [NEW PHRASE]:

        """
    def emotion(self, text, style, language_style, template_name):
            template = getattr(self, template_name)
            prompt = PromptTemplate(template=template, input_variables=["style", "text", "language_style"])
            llm_chain = LLMChain(prompt=prompt, llm=self.llm)
            return llm_chain.run({"style": style, "text": text, "language_style":language_style})


def read_text_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

emotion_map = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism", "relief", "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"]
}

def aggregate_emotion_scores(predictions, emotion_map=emotion_map):
    """
    Aggregate mean scores of related emotions based on the provided mapping and return the dominant emotion.

    :param predictions: List of predictions from the classifier.
    :param emotion_map: Dictionary mapping target styles to related prediction labels.
    :return: The dominant emotion and its mean score.
    """
    # Initialize a dictionary to hold the aggregated scores for each target style
    emotion_scores = {emotion: [] for emotion in emotion_map.keys()}  # Use lists to store scores for mean calculation

    # Count occurrences of each emotion for mean calculation
    emotion_counts = {emotion: 0 for emotion in emotion_map.keys()}

    # Iterate through each prediction and aggregate scores for the relevant target styles
    for prediction in predictions:
        for emotion, related_emotions in emotion_map.items():
            if prediction['label'] in related_emotions:
                emotion_scores[emotion].append(prediction['score'])
                emotion_counts[emotion] += 1

    # Calculate mean scores for each emotion
    
    emotion_mean_scores = {emotion: (sum(scores)/emotion_counts[emotion] if emotion_counts[emotion] > 0 else 0) for emotion, scores in emotion_scores.items()}

    # Determine the dominant emotion (emotion with the highest mean score)
    dominant_emotion = max(emotion_mean_scores, key=emotion_mean_scores.get)
    dominant_score = emotion_mean_scores[dominant_emotion]

    return dominant_emotion, dominant_score



    
def clean_text(text):
    # First, remove all double or single quotes from the start of the string
    cleaned_text = re.sub(r'^[\"\']+', '', text)
    # Next, remove all double or single quotes from the end of the string
    cleaned_text = re.sub(r'[\"\']+$', '', cleaned_text)
    return cleaned_text.strip()

def main():
    parser = argparse.ArgumentParser(description='Run Text Summarization')
    parser.add_argument('--model', type=str, help='The model name to use for summarization', default="meta-llama/Llama-2-13b-chat-hf")
    parser.add_argument('--output_file', type=str, required=False, help='The input text file for summarization')
    parser.add_argument('--template', type=str, default='second_template',
                        help='The template to use for summarization')
    args = parser.parse_args()

    template = args.template


    #roberta model
    roberta_id = "SamLowe/roberta-base-go_emotions"
    classifier = pipeline(task="text-classification", model=roberta_id, top_k=None)

    # model_id = "LeoLM/leo-mistral-hessianai-7b"
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    # med_model_id = "malhajar/meditron-7b-chat"

    # emotionr_sentiment = EmotionTransfer(args.model)
    sentiment_text = EmotionTransfer(model_id)

    sentences = [
        "The weather is clear and sunny.",
        "She finished reading the book and closed it.",
        "The cat sits on the windowsill, looking outside.",
        "He decided to take a walk in the park.",
        "They are preparing a meal in the kitchen.",
        "The museum is open to visitors from nine to five.",
        "I have a meeting scheduled for tomorrow afternoon.",
        "The train arrives at the station at six in the evening.",
        "She planted a new flower in her garden.",
        "He picked up the guitar and started tuning it."
    ]
    new_sentences = [
        "The sky is overcast, but it's not raining.",
        "He finished his coffee and set down the mug.",
        "The dog lies quietly by the fireplace, dozing off.",
        "She decided to bake cookies for the weekend.",
        "They gathered around the table to play a board game.",
        "The library offers free Wi-Fi to its patrons.",
        "My appointment is set for next Monday morning.",
        "The bus departs from the terminal every hour.",
        "He added a new book to his collection.",
        "She tuned the piano before starting her practice.",
        "A gentle breeze is blowing through the trees.",
        "The meeting concluded with a plan for the next steps.",
        "Birds are chirping early in the morning.",
        "The stars are visible in the night sky.",
        "Fresh flowers are arranged on the dining table.",
        "The path leads to a quiet, secluded beach.",
        "Homework is due by the end of the week.",
        "The painting was hung above the fireplace.",
        "A new cafe has opened down the street.",
        "They are painting the living room a light blue."
    ]

    alej_phrases = [
    "After work, I am going to the cinema with some friends",
    "They finally decided to go on vacation to Germany",
    "She will leave us soon",
    "We have to go to the gym this afternoon",
    "The professor told us that the assignment is due next week",
    "My wife had sent me a list of things to buy",
    "Our dog barks at us every time we come home",
    "This Christmas we are going to the mountain",
    "Someone ate the ice cream I left in the fridge",
    "We are expecting some customers soon",
    "My friend has bought some tickets for tonight’s game",
    "Our boss has called us in for a meeting",
    "The museum is closing tomorrow",
    "The library has removed all the fantasy books",
    "I have heard that the store closes next month",
    "She told me that she was going to be busy for the rest of the week",
    "I have found a way to finish this quickly",
    "My aunt has decided to celebrate her birthday this weekend.",
    "She was lying in the ground when I arrive",
    "The beach is completely full today",
    "I have not receive an offer for the clothes I am selling",
    "We went to the office to speak with the administrator, but no one was there.",
    "We met in the forest yesterday",
    "We spent the whole day walking around Munich",
    "My mom has bought a new toy for the cat",
    "They are taking this show off the air",
    "He told me to join for a beer tomorrow at five",
    "This coffee is hot",
    "We went to see the new Marvel movie in the theater yesterday",
    "I think England is winning the upcoming Eurocup",
    "They are playing your song next",
    "The water in the pool is colder this year",
    "The price of the food has increased in these months",
    "I have been living in here for two years now",
    "I was listening your new song when she called me",
    "This visit should take us 2 hours",
    "I am taking my old disguise to the carnival party",
    "We have an appointment to see a new house next week",
    "I got a call from my son’s school",
    "They have closed the Chinese place down the street",
    "He asked me to help him paint his room",
    "My father has already prepared the food for tonight",
    "My grandfather is telling another story in the living room."
]

    sentences = sentences + new_sentences + alej_phrases
    
    affective_labels = ["anger", "fear", "disgust", "joy", "surprise", "sadness"]

    with open(args.output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Sentence', 'Neutral_Text', 'Llama7_Text', 'Prompted_Emotion', 'Roberta_Prediction', 'Score'])

        # text = read_text_from_file(args.input_file)
        for index, sentence in enumerate(tqdm(sentences, desc="Processing sentences")):
            for style in tqdm(affective_labels, desc=f"Styles for sentence {index}", leave=False):
                response = sentiment_text.emotion(sentence, style, language_style, template)
                predictions = classifier(response)[0]
                dominant_emotion, dominant_score = aggregate_emotion_scores(predictions, emotion_map)
                writer.writerow([index, sentence, clean_text(str(response)), style, dominant_emotion, dominant_score])
                print(f"Data written for sentence {index} with prompted emotion '{style}' \n")
    # batches("output_file.csv", sentences, affective_labels, sentiment_text, template, clean_text, batch_size=8)

if __name__ == "__main__":
    main()
