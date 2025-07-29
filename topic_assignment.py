import os
import argparse
import pandas as pd
from utils import OpenAIModel
from tqdm import tqdm
import tiktoken


class ContentTopicProcessor:
    def __init__(self, args):
        self.api_key = args.API_KEY
        self.model_name = args.model_name
        self.num_samples = args.num_samples
        self.input_file = args.input_file
        self.output_file = args.output_file
        self.prompt_file = args.prompt_file
        self.openai_api = OpenAIModel(self.api_key, self.model_name, [], 2048)
        self.price_table = {'gpt-4': [0.03, 0.06], 'gpt-4o': [0.0025, 0.00125], 'text-davinci-003': [0.02, 0.02], 'gpt-3.5-turbo': [0.0015, 0.002]}

        # List of topic column names in the Excel file
        self.topic_columns = [
            'health_exp', 'public_assist', 'immigration_boarder', 'abortion',
            'LGBTQ', 'international_relation', 'domestic_crime',
            'inflation_live_exp', 'religion', 'unemployment', 'income_tax',
            'middle_class', 'state_name', 'competitor_name', 'supporter_name',
            'democracy', 'pluralist_values', 'disagreement', 'republican', 'democratic'
        ]

    def load_data(self):
        # Load the Excel file
        df = pd.read_excel(self.input_file)
        print(f"Loaded {len(df)} rows from {self.input_file}.")
        return df

    def load_prompt(self):
        # Load the prompt template from a file
        with open(self.prompt_file, 'r') as file:
            return file.read()

    def process_content(self, content, prompt_template):
        # Fill the prompt with the content
        prompt = prompt_template.replace("[[CONTENT]]", content)
        response = self.openai_api.generate(prompt)
        return response.strip()

    def count_price(self, prompts, responses):
        # Calculate estimated cost of API calls
        total_price = 0
        for prompt, response in zip(prompts, responses):
            encoding = tiktoken.encoding_for_model(self.model_name)
            input_tokens = len(encoding.encode(prompt))
            output_tokens = len(encoding.encode(response))
            price = (self.price_table[self.model_name][0] * input_tokens +
                     self.price_table[self.model_name][1] * output_tokens) / 1000
            total_price += price
        return total_price

    def process_data(self, df, prompt_template):
        # Limit to the specified number of samples (if applicable)
        if self.num_samples > 0:
            df = df.iloc[:self.num_samples].copy()

        # Prepare prompts and collect responses
        prompts = []
        responses = []

        # Initialize topic columns with zeros
        for column in self.topic_columns:
            df[column] = 0

        for idx, content in tqdm(df['text'].items(), desc="Processing rows"):
            try:
                # Generate the response
                response = self.process_content(content, prompt_template)

                # Log the response for debugging
                print(f"Row {idx}: Response: {response}")

                # Save the prompt and response
                prompts.append(prompt_template.replace("[[CONTENT]]", content))
                responses.append(response)

                # Parse topics from the response
                if response.strip().lower() == "the text does not belong to any of the given topics.":
                    print(f"Row {idx}: No topics assigned (all zeros).")
                    continue  # All topics remain 0 by default
                else:
                    assigned_topics = [topic.strip() for topic in response.split(',')]
                    print(f"Row {idx}: Parsed Topics: {assigned_topics}")  # Debug parsed topics

                    # Update topic columns
                    for topic in assigned_topics:
                        if topic in self.topic_columns:
                            df.at[idx, topic] = 1
                            print(f"Row {idx}: Set column '{topic}' to 1")
                        else:
                            print(f"Row {idx}: Topic '{topic}' not found in topic_columns.")

            except Exception as e:
                print(f"Error processing content at row {idx}: {e}")
                # Add default zeroes for all topics in case of an error
                for column in self.topic_columns:
                    df.at[idx, column] = 0

        # Calculate and print the total cost
        total_cost = self.count_price(prompts, responses)
        print(f"Estimated cost: ${total_cost:.2f}")

        return df


    def save_data(self, df):
        # Save the updated dataframe to an Excel file
        df.to_excel(self.output_file, index=False)
        print(f"Results saved to {self.output_file}.")

    def run(self):
        df = self.load_data()
        prompt_template = self.load_prompt()
        df = self.process_data(df, prompt_template)
        self.save_data(df)


def parse_args():
    parser = argparse.ArgumentParser(description="Content Topic Processor")
    parser.add_argument('--API_KEY', type=str, required=True, help="OpenAI API key.")
    parser.add_argument('--model_name', type=str, default="gpt-4", help="Model name.")
    parser.add_argument('--num_samples', type=int, default=-1, help="Number of samples to process. Use -1 for all.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to input Excel file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save output Excel file.")
    parser.add_argument('--prompt_file', type=str, required=True, help="Path to the prompt text file.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    processor = ContentTopicProcessor(args)
    processor.run()