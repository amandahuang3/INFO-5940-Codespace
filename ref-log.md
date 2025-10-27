## External Sources

### 1. SpaCy Text Splitter
**Tool/Library:** `langchain_text_splitters.spacy`  
**Description:** Used for text file segmentation via `SpacyTextSplitter` class in the text splitting part   
**Source:** https://python.langchain.com/docs/how_to/split_by_token/ 

---

### 2. RecursiveCharacterTextSplitter
**Tool/Library:** `langchain.text_splitter.RecursiveCharacterTextSplitter`  
**Description:** Used for pdf file segmentation via `RecursiveCharacterTextSplitter` class in the text splitting part  
**Source:** https://python.langchain.com/docs/how_to/recursive_text_splitter/   

## AI Assistance Log

1. **Description:** For assistance with structuring and formatting dynamic citations when model generate the respopnse. Use it because not sure how to concatenate document metatadata into citation string that will be combine in the output response.    
    **Part:** generate() function (line#149-164) & file reading (line#81-84)  
    **Assistance Source:** ChatGPT  
--- 
2. **Description:** For help with PDF upload handling that requires temporary disk storage (not in-memory) due to library limitations. Use it because not sure that in-memory files are not supported what are other way to properly handle the PDF file.   
    **Part:** PDF file handling (line #74-76)   
    **Assistance Source:** ChatGPT  

