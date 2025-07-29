import os
import argparse
import pandas as pd
from utils import OpenAIModel
from tqdm import tqdm
import tiktoken


class ContentNumberProcessor:
    def __init__(self, args):
        self.api_key = args.API_KEY
        self.model_name = args.model_name
        self.num_samples = args.num_samples
        self.input_file = args.input_file
        self.output_file = args.output_file
        self.openai_api = OpenAIModel(self.api_key, self.model_name, [], 2048)
        self.price_table = {'gpt-4': [0.03, 0.06], 'text-davinci-003': [0.02, 0.02], 'gpt-3.5-turbo': [0.0015, 0.002]}

    def load_data(self):
        # Load the Excel file
        df = pd.read_excel(self.input_file)
        print(f"Loaded {len(df)} rows from {self.input_file}.")
        return df

    def process_content(self, content):
        # OpenAI prompt to identify numeric content
        prompt = f"Does the following text contain numbers or numeric formats (e.g., 'three-star')? Respond with '1' if true, '0' otherwise.\n\n{content}"
        response = self.openai_api.generate(prompt)
        return int(response.strip()) if response.strip() in ['0', '1'] else 0

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

    def process_data(self, df):
        # Limit to the specified number of samples (if applicable)
        if self.num_samples > 0:
            df = df.iloc[:self.num_samples].copy()

        # Prepare prompts and collect responses
        prompts = []
        responses = []
        numbers_column = []
        for content in tqdm(df['content'], desc="Processing rows"):
            try:
                prompt = f"Does the following text contain numbers or numeric formats (e.g., 'three-star')? Respond with '1' if true, '0' otherwise.\n\n{content}"
                response = self.openai_api.generate(prompt)
                result = int(response.strip()) if response.strip() in ['0', '1'] else 0
                prompts.append(prompt)
                responses.append(response)
                numbers_column.append(result)
            except Exception as e:
                print(f"Error processing content: {e}")
                numbers_column.append(0)

        # Add the Numbers column using .loc to avoid SettingWithCopyWarning
        df.loc[:, 'Numbers'] = numbers_column

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
        df = self.process_data(df)
        self.save_data(df)


def parse_args():
    parser = argparse.ArgumentParser(description="Content Number Processor")
    parser.add_argument('--API_KEY', type=str, required=True, help="OpenAI API key.")
    parser.add_argument('--model_name', type=str, default="gpt-4", help="Model name.")
    parser.add_argument('--num_samples', type=int, default=-1, help="Number of samples to process. Use -1 for all.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to input Excel file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save output Excel file.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    processor = ContentNumberProcessor(args)
    processor.run()
