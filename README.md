

# Maitri AI - Period Care Recommender

[![maitri2.png](https://i.postimg.cc/1znTVMz3/maitri2.png)](https://postimg.cc/wtzFSXJC)

Maitri-AI is your Period Care Recommender. It is a personalized care suggestion tool designed to provide tailored recommendations based on the phase of the menstrual cycle. Whether you're in the menstrual phase, proliferative phase, ovulation phase, or luteal phase, this tool offers customized advice to help you take care of yourself better.

## Features

- **Personalized Care Suggestions**: Get personalized care suggestions based on the phase of your menstrual cycle and any specific issues you're facing.
- **Interactive Interface**: User-friendly interface with options to select the phase and provide additional information about symptoms.
- **Intelligent Recommendation Engine**: Powered by Langchain and Mistral AI (Mixtral-8x7B-Instruct) technologies to deliver accurate and detailed care recommendations.
- **Privacy and Confidentiality**: Your privacy is important. No personal data is stored, and all interactions are kept confidential.
  
## Technologies Used

- **Backend Language**: Python
- **AI Framework**: Langchain
- **Base Large Language Model Used**: Mixtral-8x7B-Instruct hosted by Fireworks AI
- **Web Framework**: Streamlit UI
  
## How to Use

1. **Select Menstrual Phase**: Choose the phase of your menstrual cycle from the options provided.
2. **Provide Additional Information**: Optionally, provide details about symptoms or issues you're experiencing.
3. **Get Personalized Suggestions**: Click the "Get Suggestions" button to receive personalized care recommendations tailored to your needs.

## Running Locally

To run Period Care Recommender locally on your machine, follow these steps:

### Prerequisites

1. **Python**: Ensure you have Python installed on your system. You can download it from the [official Python website](https://www.python.org/downloads/).

2. **Virtual Environment (Optional)**: It's recommended to use a virtual environment to manage dependencies. You can create one using `virtualenv` or `venv`.

### Installation

1. **Clone the Repository**: Clone the Period Care Recommender repository to your local machine using the following command:

   ```bash
   git clone https://github.com/sagnik-datta-02/period-care-recommender.git
   ```

2. **Navigate to the Directory**: Move into the Period Care Recommender directory:

   ```bash
   cd period-care-recommender
   ```

3. **Install Dependencies**: Install the required Python packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. **Environment Variables**: If applicable, create a `.env` file in the root directory and add any necessary environment variables.

### Running the Application

Once you've installed the dependencies, you can run Period Care Recommender locally:

```bash
streamlit run periodcarerecommender.py
```

The application will start running, and you can access it in your web browser by navigating to `http://localhost:8501`.

## Contributing

Contributions to Period Care Recommender are welcome! If you have ideas for improvements, feature requests, or bug reports, feel free to open an issue or submit a pull request.



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
