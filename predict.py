from model import train_model_factory, val_model_factory, predict_model_factory



class ChatBot(object):
    model = None

    def __init__(self,model_path):
        self.model=predict_model_factory(model_path)
        print('Bonjour ! Comment puis-je vous aider ?')
        

    def reply(self, input_text):
        answer=model(input_text)

        return "Salut !"

    def test_run(self,ques):
       p=self.reply(ques)
       return p