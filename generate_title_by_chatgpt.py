from revChatGPT.V1 import Chatbot
import os
import time

def request(chatbot, prompt = "昭和年代日本最流行的歌曲包括哪些？"):
    ress = []
    for data in chatbot.ask(prompt):
        response = data["message"]
        ress.append(response)
    print(response)
    return response


def arts():
    from raw_data import article_analyse
    arts = article_analyse()
    text_trimed = []
    for ss in arts:
        total_length = sum([len(s) for s in ss])
        if total_length > 1000:
            text = ''
            # NOTE: 需要一些剪裁技巧……
            max_sentence_length = int(1600 / len(ss))
            for s in ss:
                if len(s) < max_sentence_length:
                    text += s
                else:
                    text += (s[:max_sentence_length] + '…')
            text_trimed.append(text)
        else:
            text_trimed.append(''.join(ss))
    return text_trimed

def write_titles(titles):
    f = open(f'./achive/titles_generated_by_chatgpt.txt','w+', encoding="utf-8")
    for title in titles:
        f.write(f'{title}\n')
    f.close()


def run(counter = 0):
    chatbot = Chatbot(config={"access_token": os.getenv('OPENAI_TOKEN')})
    texts = arts()
    titles = []
    while True:
        # Code executed here
        prompt = f'文章の一部でタイトルを作ってください: {texts[counter]}'
        print(prompt)
        chatbot.reset_chat() # NOTE: NEED THIS TO AVOID HISTORY
        titles.append(request(chatbot, prompt))
        write_titles(titles)
        counter += 1
        print(f'{counter}\n\n')
        time.sleep(70)
    print('全部完成\n\n\n')



