# AAI-520-Final-Project: Developing a Conversational Chatbot GPT (Generative Pre-trained Transformer) and Diagloflow

![Chatbot Banner](https://github.com/bpayton0101/AAI-520-Final-Project/blob/main/Chatbot%20Images/Online-Chatbots-for-Ecommerce.png)

## Introduction

Chatbots are conversational agents  designed with the help of AI (Artificial Intelligence) software. They simulate a conversation (or a chat) with users in a natural language via messaging applications, websites, mobile apps, or phone (Irsath, 2021). Chatbots have become very popular tools ever since deep learning became popular. Thanks to deep learning, we're now able to train the bot to provide better and personalized questions, and, in the last implementation, to retain a per-user context (Byiringiro et al, 2018). 

Building a chatbot involves several key steps and can be approached in different ways. The best approach for building a chatbot depends on your specific needs and goals, and factors such as the complexity of the task and the desired level of sophistication should be taken into consideration. 

The goal of this project is to build a chatbot that can carry out multi-turn conversations, adapt to context, and handle a
variety of topics, via web or app interface where users can converse with the chatbot. This project will use the Cornell Movie Dialogs Corpus to examine generative-based chatbot architectures like Seq2Seq, Transformers, Generative Pre-trained Transformer (GPT) and deep learning. 

Project Prerequisites:
* Basic understanding of deep learning and neural networks.
* Familiarity with a deep learning framework (e.g., TensorFlow, PyTorch)
* Basic knowledge of web development (for the interface)
 
Project Phases:
* Research and Study Phase
* Data Collection and Preprocessing
* Model Design and Training
* Evaluation


## Table of Contents
1. [ Cornell Movie Dialog Corpus ](#Dataset)
2. [ Research and Study Phase  ](#Research)
3. [ Data Collection and Preprocessing  ](#Preprocessing)
4. [ Model Design and Training](#Model)
5. [ Evaluation ](#Evaluation)
6. [ Discussion and Conclusion](#dicussion)



## Dataset
The Cornell Movie Dialogue Corpus contains a metadata-rich collection of fictional conversations extracted from raw movie scripts:

- 220,579 conversational exchanges between 10,292 pairs of movie characters
- involves 9,035 characters from 617 movies
- in total 304,713 utterances
- movie metadata included:


## Research and Study Phase

During the research and study phase, we examined two chatbot architectures and compared their features and benefits before selecting a  model as the framework for our web interface.

### Model 1: Seq2Seq with Memory Networks

Seq2Seq (Sequence-to-Sequence) is a type of deep learning model that can be used to map input sequences to output sequences of varying lengths. It's a powerful technique for tasks like machine translation, text summarization, and chatbot response generation.

Sequence-to-Sequence (Seq2Seq) modeling, when paired with Long-Short-Term Memory (LSTM) units, has demonstrated significant potential in developing  conversational chatbot capable of participating in text based conversation and providing human-like responses.

Seq2Seq models are effective for tasks involving sequential data, but they can struggle with tasks that require long-term memory or reasoning over complex knowledge bases. Memory networks address these limitations by introducing a memory component that can store and retrieve information over time.

Memory network models are a class of neural networks designed to handle tasks that require remembering and reasoning over past information. While they have shown promise in various NLP tasks, including question answering and dialogue systems, they face several challenges when applied to multi-conversational chatbots:


#### 1. Long-Term Memory:
* Contextual Drift: Over long conversations, the model may struggle to maintain context and remember relevant information from earlier turns.
* Memory Capacity: Memory networks may have limitations in storing and retrieving information from a large knowledge base.
  
#### 2. Handling Complex Conversations:
* Nested References: Dealing with nested references and implicit information can be challenging for memory networks.
* Ambiguity and Polysemy: Resolving ambiguities and handling multiple meanings of words can be difficult.
  
#### 3. Efficiency:
* Computational Cost: Memory networks can be computationally expensive, especially for long conversations and large knowledge bases.
* Scalability: Scaling memory networks to handle large-scale conversational datasets can be challenging.

#### 5. Training Data:
* Quality and Quantity: Obtaining high-quality training data for multi-conversational chatbots is often difficult, as it requires capturing natural conversation flows and context.
* Bias: Biases in the training data can lead to biased and unfair responses from the chatbot.
  
#### 6. Evaluation Metrics:
* Task-Specific Evaluation: Evaluating the performance of multi-conversational chatbots requires task-specific metrics that go beyond traditional metrics like accuracy or F1-score

We observed all of these challenges when piloting the movie_lines.txt file processing with this model, so we examined the models capabilities with two publicly available datasets called:  'train_qa.txt' and 'test_qa.txt'. The results of our research of this model are cited below and the code for this model can reviewed in the Seq2Seq Deep Learning Notebook. This model was not selected as a baseline for our chatbot web interface due to various challenges with encoding and creating a suitable, clean training file for the memory network.

Accuracy: 0.503

Precision: 0.253

Recall: 0.503

F1-score: 0.3366


### Model 2: GPT with Dialogflow

GPT (Generative Pre-trained Transformer) models are designed to handle dialogue data effectively by leveraging their ability to understand and generate human-like text, and offer several advantages over traditional rule-based chatbots. GPT models are trained on massive amounts of text data, allowing them to understand and generate human-like text. This makes them more capable of engaging in natural and meaningful conversations. GPT chatbot models can help reduce the need for extensive data cleaning for large, complex, heterogeneous datasets such as the Cornell Movie Corpus. 

Dialogflow is a powerful platform for building conversational interfaces. Integrating GPT with Dialogflow can enhance the capabilities of your chatbot by Natural Language Understanding, Response Generation, and Contextual Understanding. By integrating GPT with Dialogflow, you can create more sophisticated and engaging chatbots that can handle a wider range of user queries and provide more informative and helpful responses.

* Tokenization: The input text (dialogue) is broken down into individual tokens (words or subwords).
* Embedding: Each token is represented as a numerical vector, capturing its semantic meaning.
* Encoding: The sequence of token embeddings is processed by the model's encoder, which extracts relevant information from the text.
* Decoding: The decoder generates a response based on the encoded input and the model's internal state.
* Contextual Understanding: GPT models can maintain context throughout a conversation, allowing them to provide more relevant and coherent responses.

Key factors that contribute to GPT's ability to handle dialogue data:

* Pre-training on Massive Text Datasets: GPT models are trained on massive amounts of text data, which helps them learn the nuances of human language and conversation patterns.
* Attention Mechanism: The attention mechanism in GPT models allows the model to focus on different parts of the input sequence at different times, enabling it to capture context and dependencies.
* Generative Capabilities: GPT models can generate new text, making them suitable for tasks like response generation and open-ended conversations.

GPT (Generative Pre-trained Transformer) chatbots and memory network chatbots are both powerful approaches to building conversational agents, but they have distinct characteristics and strengths. GPT's strengths of NLU, versatility, and contextual understanding outweigh the iterative manual python code requirements of Seq2Seq memory networks, and as a result, GPT with Diagloflow was selected as our chatbot model.

## Data Collection & Preprocessing
1. Data Preprocessing:

* Contraction Expansion:  Using the `contractions` library to expand contractions (e.g., "can't" to "cannot").
* Emoji Handling: Potential conversion of emoticons to text using the `emoji` library, likely for sentiment analysis (though not directly implemented in the provided code).
* Unicode Normalization: Potential normalization of Unicode characters using `unicodedata` (not directly implemented).
* Data Splitting: Utilizing `train_test_split` from scikit-learn to divide the dataset into training and validation sets.


## Model Design & Training

### Model Design

GPT (Generative Pre-trained Transformer) chatbots and memory network chatbots are both powerful approaches to building conversational agents, but they have distinct characteristics and strengths. GPT's strengths of NLU, versatility, and contextual understanding outweigh the iterative manual python code requirements of Seq2Seq memory networks, and as a result, GPT with Diagloflow was selected as our chatbot model. 

This section summarizes the design and implementation of a fine-tuned DialoGPT model for chatbot functionality. The Chaflix notebook provides the complete python code for our selected chatbot model.


2. Model Architecture:
 
* **Base Model:** The model is based on DialoGPT (likely "microsoft/DialoGPT-small" or a fine-tuned variant), a transformer-based language model suited for dialogue generation.
* **Fine-tuning:** The code performs fine-tuning on a custom dataset.
* **Gradient Checkpointing:**`model.gradient_checkpointing_enable()` is used to reduce memory consumption during training, trading compute for memory.
* **AdamW Optimizer:** AdamW optimizer is used with a learning rate of 5e-6 and weight decay set to 0.0.
* **Learning Rate Scheduler:** A linear learning rate scheduler (`get_linear_schedule_with_warmup`) is employed with a warm-up period (10% of total training steps).
* **Gradient Clipping:** To prevent exploding gradients, gradient clipping (`torch.nn.utils.clip_grad_norm_`) is implemented with a `max_norm` of 1.0.  This helps stabilize training by limiting the magnitude of gradients during backpropagation.
* **Gradient Accumulation:** Gradients are accumulated over 2 batches before updating model parameters. This simulates a larger batch size while reducing memory requirements.

### Training

3. Training
   
* **Base Model:** The model is based on DialoGPT (likely "microsoft/DialoGPT-small" or a fine-tuned variant), a transformer-based language model suited for dialogue generation. The code performs fine-tuning on a custom dataset.
* **Gradient Checkpointing:** `model.gradient_checkpointing_enable()` is used to reduce memory consumption during training, trading compute for memory.
* **AdamW Optimizer:**  AdamW optimizer is used with a learning rate of 5e-6 and weight decay set to 0.0.
* **Learning Rate Scheduler:** A linear learning rate scheduler (`get_linear_schedule_with_warmup`) is employed with a warm-up period (10% of total training steps).
* **Gradient Clipping:** To prevent exploding gradients, gradient clipping (`torch.nn.utils.clip_grad_norm_`) is implemented with a `max_norm` of 1.0.  This helps stabilize training by limiting the magnitude of gradients during backpropagation.
* **Gradient Accumulation:** Gradients are accumulated over 2 batches before updating model parameters. This simulates a larger batch size while reducing memory requirements.

### Evaluation

4. Inference and Deployment:

* **Chat Interface:** Gradio is used to create a chatbot interface.
* **Response Generation:**  The `generate_response` function tokenizes user input, concatenates it with conversation history, and generates a response using the fine-tuned model.
* **History Management:**  The chatbot maintains a conversation history for context in subsequent turns.



