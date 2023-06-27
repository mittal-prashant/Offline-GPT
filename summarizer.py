# Author: Prashant Mittal

from warnings import simplefilter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
import sys

import os
from dotenv import load_dotenv
from langchain.llms import GPT4All
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate


from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
)
from typing import List
from langchain.docstore.document import Document


# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if "text/html content not found in email" in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".xlsx": (UnstructuredExcelLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


# Taking out the warnings

load_dotenv()


embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
model_path = os.environ.get("MODEL_PATH")
model_path = os.environ.get("MODEL_PATH")
model_n_ctx = os.environ.get("MODEL_N_CTX")
model_n_batch = int(os.environ.get("MODEL_N_BATCH", 8))


user = sys.argv[1] if len(sys.argv) > 1 else ""
project = sys.argv[2] if len(sys.argv) > 2 else ""
file_arg = sys.argv[3] if len(sys.argv) > 3 else ""


# @jit(target_backend="cuda")
def main():
    file_path = f"source_documents/{user}/{project}/{file_arg}"
    # file_path = f"source_documents/logo.pptx"

    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)

    file = loader.load()

    text = ""

    for doc in file:
        text += doc.page_content

    text = text.replace("\t", " ")

    # num_tokens = llm.get_num_tokens(text)

    # print(f"This book has {num_tokens} tokens in it")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "\t"], chunk_size=3000, chunk_overlap=1000
    )
    docs = text_splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    vectors = np.array(embeddings.embed_documents([x.page_content for x in docs]))

    # Assuming 'embeddings' is a list or array of 1536-dimensional embeddings

    # Choose the number of clusters, this can be adjusted based on the book's content.
    # I played around and found ~2 was the best.
    # Usually if you have 2 passages from a book you can tell what it's about
    num_clusters = 2

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)

    ###

    # Filter out FutureWarnings
    simplefilter(action="ignore", category=FutureWarning)

    # Perform t-SNE and reduce to 2 dimensions
    tsne = TSNE(n_components=2, random_state=42, perplexity=len(file))
    reduced_data_tsne = tsne.fit_transform(vectors)

    # Create an empty list that will hold your closest points
    closest_indices = []

    # Loop through the number of clusters you have
    for i in range(num_clusters):
        # Get the list of distances from that particular cluster center
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)

        # Find the list position of the closest one (using argmin to find the smallest distance)
        closest_index = np.argmin(distances)

        # Append that position to your closest indices list
        closest_indices.append(closest_index)

    selected_indices = sorted(closest_indices)

    map_prompt = """
    You will be given a single passage of a book. This section will be enclosed in triple backticks (```)
    Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
    Your response should be at least three paragraphs and fully encompass what was said in the passage.

    ```{text}```
    FULL SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    llm = GPT4All(
        model=model_path,
        n_ctx=model_n_ctx,
        backend="gptj",
        n_batch=model_n_batch,
        verbose=False,
    )

    map_chain = load_summarize_chain(
        llm=llm, chain_type="stuff", prompt=map_prompt_template
    )

    selected_docs = [docs[doc] for doc in selected_indices]

    summary_list = []

    # Loop through a range of the lenght of your selected docs
    for i, doc in enumerate(selected_docs):
        # Go get a summary of the chunk
        chunk_summary = map_chain.run([doc])

        # Append that summary to your list
        summary_list.append(chunk_summary)


if __name__ == "__main__":
    main()
