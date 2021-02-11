# qclip

Test out [CLIP](https://openai.com/blog/clip) from OpenAI on various datasets like [ImageNetV2](https://imagenetv2.org/), [Coco](https://cocodataset.org/), and [DeepDrive](https://bdd-data.berkeley.edu/). 

## Examples

## Usage

1. Install the dependencies from `requirements.txt`

```
pip install -r requirements.txt 
```

2. To use [ImageNetv2](https://imagenetv2.org/), download the matched-frequency dataset to the `imagenetv2` directory. 

3. To use [DeepDrive](https://bdd-data.berkeley.edu/), download the Images dataset to the `bdd` directory. 

4. Run the Streamlit app

```
streamlit run clip_app.py
```
