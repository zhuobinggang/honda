from openai import OpenAI
import pyperclip
client = OpenAI()

def quest_gpt(text):
    return quest_gpt_raw(None, text)

def quest_gpt_raw(system_msg, user_msg, gpt_type = 'gpt-3.5-turbo-0125'):
    completion = client.chat.completions.create(
      model=gpt_type, # 
      messages=[
        # {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
      ],
      temperature = 0
    )
    response = completion.choices[0].message.content
    pyperclip.copy(response)
    return response
    # To clipboard
    # content = completion.choices[0].message.content
    # usage = str(completion.usage)
    # print(content)
    # print(usage)
    # # To clipboard
    # # NOTE: 自动提取command
    # copied = False
    # for line in content.split('\n'):
    #     line = line.lower()
    #     if line.startswith('next action:'):
    #         text_to_paste = line.replace('next action:', '').strip()
    #         pyperclip.copy(f"c('{text_to_paste}')")
    #         print(f'COMMAND GOT: {text_to_paste}')
    #         copied = True
    # if not copied:
    #     pyperclip.copy('')
    #     print(f'BE CAREFULL!')
    # dic = {'response': content, 'usage': usage}
    # return completion, dic
