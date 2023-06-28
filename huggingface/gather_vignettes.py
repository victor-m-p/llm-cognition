'''
VMP 2023-06-27:
just gathering our own vignettes.
'''
import json
sentence_dict = {'vignette_1': "Linda is thirty, and works as a lawyer at a big firm. Although Linda is successful and leads a comfortable life she does not feel fulfilled. Linda wants to make an impact and help make the world a better place. She sets a goal of helping a significant number of people in the third world, in ten years.",
                 'vignette_2': "Robert is a young man living in a poor neighborhood where crime is part of everyday life, and it seemed like you needed to be tough to get ahead. Robert is bullied and always felt like an outsider, being more interested in reading books and theater than in money and girls. Robert currently has a job that he is not passionate about, but he really dreams about making it as an actor. He sets a goal of being a professional actor 10 years from now.",
                 'vignette_3': "James is a professional online video streamer in his mid twenties. He works from home, and enjoys the freedom and the online community. However, working from home with online content sometimes makes him feel isolated. James dreams of meeting a romantic partner that he can share his life with. He sets a goal of having started his own family 10 years from now.",
                 'vignette_4': "Mary is in her early forties and lives with her family in a suburb. She has always struggled with her weight and she is a heavy smoker. Last night Mary saw a documentary which focused on the consequences of an unhealthy lifestyle. This scared Mary, and she decided that now is the time to change. Mary sets a goal of being in good shape and smoke-free in 10 years.",
                 'vignette_5': "Simon has just turned 20. He has just started college, and feels he is thoroughly mediocre. He fantasizes about becoming famous and living amongst the jet set. He sets an ambitious goal of being a recognizable face and a multi-millionaire 10 years from now.",
                 'vignette_6': "Jack is 12 years old and obsessed with football. He has moved teams, from the small league in his home-town, to one of the big teams in a larger nearby city. Although the competition is hard, Jack has the respect of his coach and is often selected for a key role in big games. Jack dreams of becoming a professional player. He sets an ambitious goal of playing professionally in 10 years.",
                 'vignette_7': "Maryam is 10 years old and lives with her family in India. She is the smartest kid in her class, and especially likes to play with numbers. While the other kids like playing, she dreams of something bigger. She sets the ambitious goal of studying mathematics in at a big university 10 years from now.",
                 'vigennte_8': "Mary is 40 years old, and has a seven year old daughter, Eve, who is struggling at school and in life. Eve has difficulties making friends at school, and is not doing as well as she could be given her innate abilities. Mary sets a ten-year goal of helping her daughter grow into a happy and thriving teenager.",
                 'vignette_9': "Justin is 40 years old, and lives in bad neighboorhood in a city that has seen better days. He believes in his neighboorhood, however. He sets a ten-year goal of making his neighboorhood a better, safer, and more prosperous place for people.",
                 'vignette_10': "Sam is 30 years old, and has a close friend, Harry, who is struggling in life. Harry has problems with addiction, and often feels lonely and unappreciated. Harry wants to get better, but is not doing as well as he could be given his innate abilities. Sam sets a ten-year goal of helping Harry become a happier and more thriving person.",
                 'vignette_11': "Jackson is 20 years old. He has grown up in a very religious family, with a father who exercises strong control over him and his brother. His older brother has achieved success and left the household, but Jackson struggles to do the same. He sets a ten-year goal of discovering who he really is, and what he really needs and desires in life.",
                 'vignette_12': "Abraham is 60 years old. He is a very religious man, who takes his moral duties very seriously. Things that most people consider normal behavior, he considers to be sinful. However, he finds himself increasingly disatisfied by how his community's religious leader tells him he ought to live. He is not sure whether he should continue to follow his religious prescriptions so closely, or, conversely, whether he should relax his standards. He sets a ten-year goal of understanding himself, and what God wants from him, better.",
                 'vignette_13': "Alexandra is 30 years old. She has been married for many years, but is unhappy with her husband. She also has children, who she loves very much. She works as a personal assistant in a large corporation, where she is often in contact with men who she finds more interesting, and more exciting, than her husband. She sets herself a ten-year goal of finding a relationship with someone like that.",
                 'vignette_14': "Emily is 40 years old. She works at a large accounting firm, where she is paid very well. Her job provides her with financial security and helps support her family. As she rises up the ranks, she realizes that the firm is successful, in part, because it cuts corners and is not entirely truthful in the reports it releases. Succeeding in the company going forward will likely involve navigating these ethical tradeoffs. She sets herself the ten-year goal of becoming CEO"
}
with open('../data/data_input/vignettes.json', 'w') as f:
    json.dump(sentence_dict, f)