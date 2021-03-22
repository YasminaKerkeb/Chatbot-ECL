from tkinter import *
from tkinter import scrolledtext
from predict import ChatBot
from tkinter import LEFT,RIGHT,TOP,BOTTOM
from PIL import Image, ImageTk
#Calling Class for chat prediction
MODEL_PATH="seq2seq-2021-03-18-03-58-16.pt"
MODE="retrieve"
ob = ChatBot(MODEL_PATH,MODE)

#main display chat window 
window = Tk()
window.title("ChatBot Centrale")
window.geometry('650x550')


# photo = PhotoImage(file = "img/logo_ecl.gif")
# w = Label(window, image=photo)
# w.pack()

#top frame to display the chat history
frame1 = Frame(window, class_="TOP")


frame1.pack(expand=True, fill=BOTH)

#text area with scroll bar
textarea = Text(frame1, state=DISABLED)
vsb = Scrollbar(frame1, takefocus=
                0, command=textarea.yview)
vsb.pack(side=RIGHT, fill=Y)
textarea.pack(side=RIGHT, expand=YES, fill=BOTH)
textarea["yscrollcommand"]=vsb.set

#bottom frame to display current user question text box 
frame2 = Frame(window, class_="Chatbox_Entry")
frame2.pack(fill=X, anchor=N)

lbl = Label(frame2, text="User : ")
lbl.pack(side=LEFT)
 

def bind_entry(self, event, handler):
    txt.bind(event, handler)

def clicked(event): 
    #to automate the scrollbar action downward according to the text
    relative_position_of_scrollbar = vsb.get()[1]
    res =txt.get() 
    #function call
    if res not in ["Non"]:
        ans = ob.test_run(res)
        pr="Human : " + res + "\n" + "ChatBot : " + ans + "\n"
        #the state of the textarea is normalto write the text to the top area in the interface
        textarea.config(state=NORMAL)
        textarea.insert(END,pr)
        pr="ChatBot : " + "Etes vous satisfait de cette réponse ?" + "\n"
        textarea.insert(END,pr)
    else:
        topic, ans=ob.get_topics_data(res)
        print(ans)
        pr="Human : " + res + "\n" + "ChatBot : Résultat pour le sujet " + topic+":\n" +ans + "\n"
        #the state of the textarea is normalto write the text to the top area in the interface
        textarea.config(state=NORMAL)
        textarea.insert(END,pr)

    #it is again disabled to avoid the user modifications in the history
    textarea.config(state=DISABLED)
    txt.delete(0,END)
    if relative_position_of_scrollbar == 1:
        textarea.yview_moveto(1)
    txt.focus()

txt = Entry(frame2,width=70)
txt.pack(side=LEFT,expand=YES, fill=BOTH)
txt.focus()
txt.bind("<Return>", clicked)

window.mainloop()