# SlideVQA
This includes an SlideVQA dataset of Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku Hasegawa, Itsumi Saito, and Kuniko Saito. "A Dataset for Document Visual Question Answering on Multiple Images". In Proc. of AAAI. 2023.

> We introduce a new document VQA dataset, SlideVQA, for tasks wherein given a slide deck composed of multiple slide images and a corresponding question, a system selects a set of evidence images and answers the question.


# Software installation
```
pip install -r requirements.txt
```
For users who wants to extract OCR with Tesseract, please install [Google Tesseract OCR](https://github.com/tesseract-ocr/tesseract).


# Get Started
## 1. Download slide images
Download 2,619 slide decks from [SlideShare](https://www.slideshare.net/). Each deck is composed of 20 slide images.
```
python download_slides_slideshare.py --target_dir TARGET_DIR --split SPLIT --sleep_time 1
```

## 2. OCR (Google Cloud Vision API)
Google Cloud Vision API is a paid OCR software, and we used the OCR resutls obtained from this OCR software in our main experiments.

Before running OCR scripts, you should obtain an API key through the [Google Cloud Platform](https://cloud.google.com/). To get one visit the [link](https://cloud.google.com/vision/docs/quickstart)
```
python extract_ocr_visionAPI.py --image_dir IMAGE_DIR --save_dir SAVE_DIR --split SPLIT
```

## 2. OCR (Tesseract)
Tesseract is a free OCR softwware. Our script uses [pytessract](https://github.com/madmaze/pytesseract), which is the wrapper for Google's Tesseract-OCR Engine.
```
python extract_ocr_tesseract.py --image_dir IMAGE_DIR --save_dir SAVE_DIR --split SPLIT
``` 

# Dataset Format
SlideVQA provides annotated 14,484 QA pairs and 890,945 bounding boxes.

## QA format
<pre>
   {
      "deck_name": slide deck name,
      "deck_url": slide deck URL in slideshare,
      "image_urls": URL list of slide deck in slideshare,
      "qa_id": identification of the QA sample,
      "question": question,
      "answer": answer,
      "arithmetic_expression": arithmetic expression to derive the answer,
      "evidence_pages": evidence pages (1 - 20) to answer the question,
      "reasoning_type": reasoning type,
      "answer_type": answer type,
    }
</pre>

## Bounding boxes format
<pre>
    {
      "deck_name": slide deck name,
      "deck_url": slide deck URL in slideshare,
      "image_urls": URL list of slide deck in slideshare,
      "category": category name of slide deck defined in slideshare,
      "bboxes": [
                  {
                    "bbox_id": identification of the bounding box,
                    "class": class name of the bounding box,
                    "bbox": [x1, y1, w, h]
                  }
                ]
    }
</pre>


# Evaluate

With the prediction and ground-truth results, you can get evaluation results on all evaluation tasks.

```
python evaluate.py --qa_preds_file QA_PREDICTIONS_FILE_NAME --es_preds_file ES_PREDICTIONS_FILE_NAME --gts_file TARGETS_FILE_NAME
```


# Citation
You can cite it as follows:
```bibtex
@inproceedings{SlideVQA2023,
  author    = {Ryota Tanaka and
               Kyosuke Nishida and
               Kosuke Nishida and
               Taku Hasegawa and
               Itsumi Saito and
               Kuniko Saito},
  title     = {SlideVQA: A Dataset for Document Visual Question Answering on Multiple Images},
  booktitle = {AAAI},
  year      = {2023}
}
```

If you have any questions about the paper and repository, feel free to contact Ryota Tanaka (ryouta.tanaka.rg[at]hco.ntt.co.jp) or open an issue!
