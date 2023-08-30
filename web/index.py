import gradio
from reviewer.reviewer import Reviewer
from utils import Console

def main(api_key, host, research_fields, review_format, pdf_file, language):
    if not pdf_file:
        return "请上传论文PDF文件！", ""
    if not api_key:
        return "请填写API-key！", ""
    if not research_fields:
        return "请填写论文的研究方向！", ""
    try:
        pdf_file_path = pdf_file.name
        reviewer = Reviewer(api_key, host, review_format, research_fields, pdf_file_path, language)
    except Exception as e:
        Console.error_bh("出现错误：" + str(e))
        return "出现错误：" + str(e), ""
    result,_  = reviewer.review()
    return result, ""

def run():
    ########################################################################################################
    # 标题
    title = "🤖ChatReviewer🤖"
    # 描述

    description = '''
    <strong>ChatReviewer是一款基于ChatGPT-3.5的API开发的智能论文分析与建议助手。</strong>其用途如下：
    ⭐️对论文的优缺点进行快速总结和分析，提高科研人员的文献阅读和理解的效率，紧跟研究前沿。
    ⭐️对自己的论文进行分析，根据ChatReviewer生成的改进建议进行查漏补缺，进一步提高自己的论文质量。
    ### 使用方式：
    #### 1. Share Token 方式(推荐)
    1. 官方 Chatgpt 登录，然后访问 [这里](http://chat.openai.com/api/auth/session) 拿 Access Token。Access Token 有效期 14 天，期间访问不需要梯子。这意味着你在手机上也可随意使用。
    2. 到 https://ai.fakeopen.com/token 去生成一个fk-开头的Share Token。
    3. 将Share Token填入 API 输入框中。
    4. 上传论文PDF文件，然后点击Submit。
    #### 2. 官方 API 方式
    1. 点击 [这里](https://platform.openai.com/account/api-keys) 申请一个OpenAI API Key(sk-xxxx)
    2. 填入 API-key 输入框中。
    3. 上传论文PDF文件，然后点击Submit。

    **两种方式选一种即可，即填了 API Key 就不要填 Share Token，填了 Share Token 就不要填 API Key。**
    原项目 [Github](https://github.com/nishiwen1214/ChatReviewer)
    '''

    # 创建Gradio界面
    inp = [
        gradio.components.Textbox(label="请输入你的API-key(sk/fk开头的字符串)",
                            type='password'),
        gradio.components.Textbox(label="API-域名(替换 https://api.openai.com/)",
                            value="https://ai.fakeopen.com/",
                            type='text'),
        gradio.components.Textbox(label="请输入论文的研究方向",
                            type='text',
                            value="computer science, artificial intelligence and reinforcement learning"),
        gradio.components.Textbox(lines=5,
            label="请输入特定的分析要求和格式(否则为默认格式)",
            value="""* Overall Review
    Please briefly summarize the main points and contributions of this paper.
    xxx
    * Paper Strength
    Please provide a list of the strengths of this paper, including but not limited to: innovative and practical methodology, insightful empirical findings or in-depth theoretical analysis,
    well-structured review of relevant literature, and any other factors that may make the paper valuable to readers. (Maximum length: 2,000 characters)
    (1) xxx
    (2) xxx
    (3) xxx
    * Paper Weakness
    Please provide a numbered list of your main concerns regarding this paper (so authors could respond to the concerns individually).
    These may include, but are not limited to: inadequate implementation details for reproducing the study, limited evaluation and ablation studies for the proposed method,
    correctness of the theoretical analysis or experimental results, lack of comparisons or discussions with widely-known baselines in the field, lack of clarity in exposition,
    or any other factors that may impede the reader's understanding or benefit from the paper. Please kindly refrain from providing a general assessment of the paper's novelty without providing detailed explanations. (Maximum length: 2,000 characters)
    (1) xxx
    (2) xxx
    (3) xxx
    * Questions To Authors And Suggestions For Rebuttal
    Please provide a numbered list of specific and clear questions that pertain to the details of the proposed method, evaluation setting, or additional results that would aid in supporting the authors' claims.
    The questions should be formulated in a manner that, after the authors have answered them during the rebuttal, it would enable a more thorough assessment of the paper's quality. (Maximum length: 2,000 characters)
    *Overall score (1-10)
    The paper is scored on a scale of 1-10, with 10 being the full mark, and 6 stands for borderline accept. Then give the reason for your rating.
    xxx"""
        ),
        gradio.components.File(label="请上传论文PDF文件(请务必等pdf上传完成后再点击Submit！)",type="file"),
        gradio.components.Radio(choices=["English", "Chinese", "French", "German","Japenese"],
                            value="English",
                            label="选择输出语言"),
    ]
    chat_reviewer_gui = gradio.Interface(fn=main,
                                    inputs=inp,
                                    outputs=[gradio.Textbox(lines=35, label="分析结果"), gradio.Textbox(lines=2, label="资源统计")],
                                    title=title,
                                    description=description)

    # Start server
    chat_reviewer_gui.launch(quiet=True, show_api=False)
