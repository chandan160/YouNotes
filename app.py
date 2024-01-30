import streamlit as st
from transformers import pipeline, AutoTokenizer
from pytube import extract, YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import base64


# Model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# Extracting transcript from video URL
def get_ts(url):
    video_id = extract.video_id(url)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    ARTICLE = " ".join(item["text"] for item in transcript)
    return ARTICLE

# Extracting the Video title
def get_title(url):
    yt = YouTube(url)
    return yt.title


# Splitting the transcript
def overlap_split(text, split_size, overlap_size):
    
    # Initialize an empty list to store the split segments
    segments = []

    # Start splitting the text with the specified overlap
    start = 0
    end = len(text)

    while start < end:
        # Determine the end index for the current segment
        segment_end = min(start + split_size, end)

        # Append the current segment to the list
        segments.append(text[start:segment_end])

        # Move the start index forward by overlap size
        start += split_size - overlap_size

    return segments


# Generating the summary

def summarize_article(url, chunk_size=1000, max_tokens_factor=0.7):

    ARTICLE = get_ts(url)
    if len(ARTICLE) > 1100:
        # Split the article into smaller parts
        chunks = overlap_split(ARTICLE, chunk_size, overlap_size = 10)
        
        # Initialize summarizer and tokenizer
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

        # Generate summaries for each chunk
        summaries = []
        for chunk in chunks:
            # Calculate max_length dynamically based on the token count of the chunk
            max_tokens = max_tokens_factor * tokenizer.model_max_length
            max_length = min(len(tokenizer.encode(chunk)), max_tokens) - 2

            # Calculate min_length (smaller than max_length)
            min_length = min(50, max_length - 5)

            # Generate summary for the chunk
            summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            summaries.append(summary[0]['summary_text'])

        # Merge all summaries into one
        merged_summary = ' '.join(summaries)
        return merged_summary
    else:
        # If the article is within the desired length, summarize it directly
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        return summarizer(ARTICLE, do_sample=False)




#streamlit code
    

def main():
    
    # Changing the page configuration
    st.set_page_config(
    page_title="YouNotes",
    page_icon="img/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        
        "About": """
            ## YouNotes
            
            
            This is an AI app which takes the URL of a youtube video as input and outputs a concise summary of that video.
        """
    }
)
    
    # Inject custom CSS for glowing border effect
    st.markdown(
        """
        <style>
        .cover-glow {
            width: 100%;
            height: auto;
            padding: 3px;
            box-shadow: 
                0 0 5px #330000,
                0 0 10px #660000,
                0 0 15px #990000,
                0 0 20px #CC0000,
                0 0 25px #FF0000,
                0 0 30px #FF3333,
                0 0 35px #FF6666;
            position: relative;
            z-index: -1;
            border-radius: 30px;  /* Rounded corners */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Function to convert image to base64
    def img_to_base64(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    # Load and display sidebar image with glowing effect
    img_path = "img/logo.png"
    img_base64 = img_to_base64(img_path)
    st.sidebar.markdown(
        f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    st.sidebar.markdown("""
        ### Critical points requiring attention:
        - This app can generate summary for only that videos which are in english language and have english captions.
        - It's essential to note that the generated summaries may contain occasional errors, urging users to approach the results with carefullness.
        """)

    # Applying the style to the title
    st.markdown(
        f'<h1 style="font-family: Bodoni; color: red; text-align: center;">YouNotes ✍️</h1>',
        unsafe_allow_html=True
    )


    # Adjusting the height of the text area for URL input
    url = st.text_input(" ", placeholder="Enter the URL")

    if url:
        quoted_url = f'"{url}"'
    if st.button("Summarize"):
        title = get_title(quoted_url)
        st.info(f'Summarization of the video with title "{title}"')
        summary = summarize_article(quoted_url)
        st.success(summary)

                

if __name__ == '__main__':
    main()
    
