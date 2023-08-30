import os
import re
import datetime
import time
import openai, tenacity
import argparse
from tiktoken_ext import openai_public
import tiktoken
from paper.get_paper_from_pdf import Paper
import jieba
from utils import Console
import constants

def contains_chinese(text):
    for ch in text:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def validateTitle(title):
    # 修正论文的路径格式
    rstr = r"[\/\\\:\*\?\"\<\>\|]" # '/ \ : * ? " < > |'
    new_title = re.sub(rstr, "_", title) # 替换为下划线
    return new_title

# 定义Reviewer类
class Reviewer:
    # 初始化方法，设置属性
    def __init__(self, api_key, host, review_format, research_fields, paper, language):
        self.review_format = review_format
        self.paper = Paper(path=paper)
        self.language = language
        self.research_fields = research_fields
        self.max_token_num = 4096
        self.encoding = tiktoken.get_encoding("gpt2")
        openai.api_key = api_key
        openai.api_base = f"{host}/v1"

    def prepare(self):
        text = ''
        text += 'Title: ' + self.paper.title + '. '
        text += 'Abstract: ' + self.paper.section_texts['Abstract']
        text_token = len(self.encoding.encode(text))
        if text_token > self.max_token_num/2 - 800:
            input_text_index = int(len(text)*((self.max_token_num/2)-800)/text_token)
            text = text[:input_text_index]
        messages = [
            {"role": "system",
             "content": f"You are a professional reviewer in the field of {self.research_fields}. "
                        f"I will give you a paper. You need to review this paper and discuss the novelty and originality of ideas, correctness, clarity, the significance of results, potential impact and quality of the presentation. "
                        f"Due to the length limitations, I am only allowed to provide you the abstract, introduction, conclusion and at most two sections of this paper."
                        f"Now I will give you the title and abstract and the headings of potential sections. "
                        f"You need to reply at most two headings. Then I will further provide you the full information, includes aforementioned sections and at most two sections you called for.\n\n"
                        f"Title: {self.paper.title}\n\n"
                        f"Abstract: {self.paper.section_texts['Abstract']}\n\n"
                        f"Potential Sections: {self.paper.section_names[2:-1]}\n\n"
                        f"Follow the following format to output your choice of sections:"
                        f"{{chosen section 1}}, {{chosen section 2}}\n\n"},
            {"role": "user", "content": text},
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        result = ''
        for choice in response.choices:
            result += choice.message.content
        print(result)
        return result.split(',')

    def review(self):
        htmls = []
        sections_of_interest = self.prepare()
        # extract the essential parts of the paper
        text = ''
        text += 'Title:' + self.paper.title + '. '
        text += 'Abstract: ' + self.paper.section_texts['Abstract']
        intro_title = next((item for item in self.paper.section_names if 'ntroduction' in item.lower()), None)
        if intro_title is not None:
            text += 'Introduction: ' + self.paper.section_texts[intro_title]
        # Similar for conclusion section
        conclusion_title = next((item for item in self.paper.section_names if 'onclusion' in item), None)
        if conclusion_title is not None:
            text += 'Conclusion: ' + self.paper.section_texts[conclusion_title]
        for heading in sections_of_interest:
            if heading in self.paper.section_names:
                text += heading + ': ' + self.paper.section_texts[heading]
        chat_review_text, _ = self.chat_review(text=text)
        htmls.append(chat_review_text)
        export_text = "\n".join(htmls)
        return export_text, self.paper.title

    def insert_sentence(self, text, sentence, interval):
        lines = text.split('\n')
        new_lines = []

        for line in lines:
            if contains_chinese(line):
                words = list(jieba.cut(line))
                separator = ''
            else:
                words = line.split()
                separator = ' '

            new_words = []
            count = 0

            for word in words:
                new_words.append(word)
                count += 1

                if count % interval == 0:
                    new_words.append(sentence)

            new_lines.append(separator.join(new_words))

        return '\n'.join(new_lines)

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
                    stop=tenacity.stop_after_attempt(5),
                    reraise=True)
    def chat_review(self, text):
        text_token = len(self.encoding.encode(text))
        if text_token > self.max_token_num/2 - 800:
            input_text_index = int(len(text)*((self.max_token_num/2)-800)/text_token)
            text = text[:input_text_index]
        input_text = "This is the paper for your review:" + text
        messages=[
                {"role": "system", "content": "You are a professional reviewer. Now I will give you a paper. You need to give a complete review opinion according to the following requirements and format:"+ self.review_format +" Please answer in {}.".format(self.language)},
                {"role": "user", "content": input_text},
        ]
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7
            )
            result = ''
            for choice in response.choices:
                result += choice.message.content
            result = self.insert_sentence(result, '**Generated by ChatGPT, no copying allowed!**', 50)
            result += "\n\n⚠伦理声明/Ethics statement：\n--禁止直接复制生成的评论用于任何论文审稿工作！\n--Direct copying of generated comments for any paper review work is prohibited!"
            usage = response.usage.total_tokens
        except Exception as e:
        # 处理其他的异常
            result = "⚠：非常抱歉>_<，生了一个错误："+ str(e)
            usage  = 'xxxxx'
        print("********"*10)
        print(result)
        print("********"*10)
        return result, usage



def export_to_markdown(text, file_name, mode='w'):
        # 使用markdown模块的convert方法，将文本转换为html格式
        # html = markdown.markdown(text)
        # 打开一个文件，以写入模式
        with open(file_name, mode, encoding="utf-8") as f:
            # 将html格式的内容写入文件
            f.write(text)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key",
                        type=str, default='',
                        help="openai api key",
                        required=True)
    parser.add_argument("--host",
                        type=str,
                        help="openai api host")
    parser.add_argument("--paper_path",
                        type=str, default='',
                        help="path of papers",
                        required=True)
    parser.add_argument("--research_fields",
                        type=str,
                        default='computer science, artificial intelligence and reinforcement learning', help="the research fields of paper")
    parser.add_argument("--review_format",
                        type=str,
                        default=constants.default_review_format,
                        help="review format. value can be a file path or a string")
    parser.add_argument("--output_file_format",
                        type=str,
                        default='txt',
                        help="output file format")
    parser.add_argument("--language",
                        type=str,
                        default="english",
                        help="output lauguage, english or chinese")
    args = parser.parse_args()
    if not args.paper_path:
        Console.error_bh("Please specify the path of the paper!")
        exit(1)
    if not args.api_key:
        Console.error_bh("Please specify the api key!")
        exit(1)

    host = "https://api.openai.com/"
    if args.host:
        host = f"{args.host}"
        Console.info_bh("Using custom host: {}".format(args.host))
    review_format = args.review_format
    if os.path.exists(review_format):
        with open(review_format, 'r', encoding='utf-8') as f:
            review_format = f.readlines(review_format)
    # 开始判断是路径还是文件：
    # paper_list = []
    # if args.paper_path.endswith(".pdf"):
    #     paper_list.append(Paper(path=args.paper_path))
    # else:
    #     for root, dirs, files in os.walk(args.paper_path):
    #         print("root:", root, "dirs:", dirs, 'files:', files) #当前目录路径
    #         for filename in files:
    #             # 如果找到PDF文件，则将其复制到目标文件夹中
    #             if filename.endswith(".pdf"):
    #                 paper_list.append(Paper(path=os.path.join(root, filename)))
    # print("------------------paper_num: {}------------------".format(len(paper_list)))
    # [print(paper_index, paper_name.path.split('\\')[-1]) for paper_index, paper_name in enumerate(paper_list)]
    start_time = time.time()
    reviewer = Reviewer(api_key=args.api_key,
                        host=host,
                        review_format=review_format,
                        paper=args.paper_path,
                        research_fields=args.research_fields,
                        language=args.language)
    export_text, paper_title = reviewer.review()
    # 将审稿意见保存起来
    date_str = str(datetime.datetime.now())[:13].replace(' ', '-')
    try:
        export_path = os.path.join('./', 'output_file')
        os.makedirs(export_path)
    except:
        pass
    file_name = os.path.join(export_path, f"{date_str}-{validateTitle(paper_title)}.md")
    export_to_markdown(export_text, file_name=file_name, mode="w")
    print(export_text)
    print("review time:", time.time() - start_time)

