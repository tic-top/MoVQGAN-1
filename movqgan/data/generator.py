"""
Generate simple text images
"""

import fitz
import random
import string
import math
from PIL import Image
from io import BytesIO

# import traceback


def build_text_img(
    text,
    min_ratio=1 / 8,
    max_ratio=8,
    min_height=64,
    max_height=1536,
    max_edge_dist=0.1,
    random_text=True,
    need_font_size=False,
    align=0,
    cur_font=None,
):
    """
    output:
        text_extract, bboxs, pix, (font_size)
    """
    assert (
        min_ratio <= 1 <= max_ratio
    ), f"min_ratio: {min_ratio}, max_ratio: {max_ratio}"
    # newer will be used later
    h_w_ratio = random.choice(
        [random.uniform(min_ratio, 1), random.uniform(1, max_ratio)]
    )
    # h_w_ratio = random.uniform(min_ratio, max_ratio)
    height = random.randint(min_height, max_height)
    width = max(max_ratio * min_height, int(height / h_w_ratio))
    edge_dist = random.uniform(0, max_edge_dist)
    q = (height * width / (2048 * 16 * 16)) ** 0.5

    page_width, page_height = width, height
    fields = text.split("\n")
    paras = []
    for field in fields:
        if len(field.strip()) == 0:
            continue
        all_punc = True
        for char in field:
            if char not in string.punctuation:
                all_punc = False
                break
        if all_punc:
            continue
        if random_text == True:
            char_list = list(field)
            random.shuffle(char_list)
            field = "".join(char_list)
        paras.append(field)

    if len(paras) > 10:
        start = random.randint(0, len(paras) - 10)
        paras = paras[start : start + 10]

    text_extract, bboxs, pix = None, None, None

    previous_size = 70
    finish = False
    while finish == False:
        try:
            if previous_size < 11:
                assert False, f"jump bad image"
            font_size = random.uniform(9, previous_size - 1)
            previous_size = font_size
            doc = fitz.open()
            page = doc.new_page(width=width, height=height)
            if cur_font is not None:
                page.insert_font(fontfile=cur_font, fontname="F0")
            x0 = edge_dist * page_width
            x1 = page_width - x0
            y0 = edge_dist * page_height
            y1 = page_height - y0
            assert x1 > x0 and y1 > y0
            rect = fitz.Rect(x0, y0, x1, y1)

            artical = "\n\n".join(paras)
            max_artical_length = 500
            start_pos = 0
            end_pos = min(max_artical_length, len(artical))

            rc = -1
            cnt = 1
            while rc < 0:
                assert (
                    start_pos < end_pos and len(artical[start_pos:end_pos]) != 0
                ), f"start_pos: {start_pos}, end_pos: {end_pos}, artical:{artical[start_pos:end_pos]}"
                if random.uniform(0, 1) > 0.5:
                    color = None
                else:
                    color = (
                        random.uniform(0, 1),
                        random.uniform(0, 1),
                        random.uniform(0, 1),
                    )

                rc = page.insert_textbox(
                    rect,
                    artical[start_pos:end_pos].strip(),
                    fontsize=int(font_size * q),
                    fontname=(
                        "Times-Roman" if cur_font is None else "F0"
                    ),  # a PDF standard font
                    fontfile=None,  # could be a file on your system
                    align=0,
                    color=color,
                )  # 0 = left, 1 = center, 2 = right
                if rc < 0:
                    end_pos -= cnt

            text_extract = []
            bboxs = []
            for block in page.get_text("dict")["blocks"]:
                for line in block["lines"]:
                    cur_line = []
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if len(text) == 0:
                            continue
                        cur_line.append(text)

                    if len(cur_line) == 0:
                        continue
                    cur_line = " ".join(cur_line).strip()
                    x0, y0, x1, y1 = (
                        line["bbox"][0] / width,
                        line["bbox"][1] / height,
                        line["bbox"][2] / width,
                        line["bbox"][3] / height,
                    )

                    assert 0 <= x0 <= 1, f"x0: {x0}"
                    assert 0 <= y0 <= 1, f"y0: {y0}"
                    assert 0 <= x1 <= 1, f"x1: {x1}"
                    assert 0 <= y1 <= 1, f"y1: {y1}"
                    bboxs.append([x0, y0, x1, y1])

                    if len(cur_line) == 0:
                        continue
                    text_extract.append(cur_line)

            pix = page.get_pixmap()
            pix_bytes = pix.tobytes(output="png")
            pix = Image.open(BytesIO(pix_bytes)).convert("RGB")
            doc.close()
            finish = True
        except Exception as e:
            doc.close()
            # print(f"--------------exception:{e}----------------")
            # traceback.print_exc()

    text_extract_check = "".join(" ".join(text_extract).split())
    gt_text = artical[start_pos:end_pos].strip()
    gt_text_check = "".join(gt_text.split())
    assert (
        text_extract_check == gt_text_check
    ), f"{text_extract_check} != {gt_text_check}"

    if need_font_size:
        return {
            "lines": text_extract,
            "bboxs": bboxs,
            "image": pix,
            "font_size": int(previous_size * q),
        }

    return {"lines": text_extract, "bboxs": bboxs, "img": pix}