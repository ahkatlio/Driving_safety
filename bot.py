import speech_recognition as sr
import pyttsx3
import pywhatkit
import datetime
import wikipedia
import pyjokes

botName = 'kit'
listener = sr.Recognizer()
engine = pyttsx3.init()
engine.say("Hello, i am"+" "+botName+"!")
engine.say("How can i help you?")
engine.runAndWait()

def talk(text):
    engine.say(text)
    engine.runAndWait()


def take_command():
    try:
        with sr.Microphone() as source:
            print('listening....')
            voice = listener.listen(source)
            command = listener.recognize_google(voice)
            command = command.lower()
            if botName in command:
                command = command.replace(botName, '')
                print(command)


    except:
        command = "no"
    return command

def run_bot():
    command = take_command()
    #print(command)
    if 'play' in command:
        song = command.replace('play','')
        talk('playing '+ song)
        pywhatkit.playonyt(song)
        pass

    elif 'say' in command:
        command = command.replace('say','')
        talk(command)
        pass

    elif 'location' in command:
        pywhatkit.search('current location')
        pass

    elif 'where am i' in command:
        pywhatkit.search('current location')
        pass

    elif 'time' in command:
        time = datetime.datetime.now().strftime('%I %M %p')
        talk('Current time is'+ time)
        #print(time)
        pass

    elif 'tell me about' in command:
        wi = command.replace('tell me about','')
        info = wikipedia.summary(wi, 1)
        #print(info)
        talk(info)
        pass

    elif 'do you know about' in command:
        wi = command.replace('do you know about','')
        info = wikipedia.summary(wi, 1)
        #print(info)
        talk(info)
        pass

    elif 'who is' in command:
        wi = command.replace('who is','')
        info = wikipedia.summary(wi, 1)
        #print(info)
        talk(info)
        pass

    elif 'where is' in command:
        wi = command.replace('where is','')
        pywhatkit.search(wi)
        pass

    elif 'what is' in command:
        wi = command.replace('what is','')
        info = wikipedia.summary(wi, 1)
        #print(info)
        talk(info)
        pass

    elif 'joke' in command:
        talk(pyjokes.get_joke())
        pass

    elif 'laugh' in command:
        talk("ha ha ha ha")
        pass


    elif 'how are you' in command:
        talk("I'm fine, Thank you, And you?")
        pass

    elif "i am fine" in command:
        talk("i am happy to hear that")
        pass

    elif "no" in command:
        pass

    else:
        talk("Can you say it again?")
        pass
while True:
    run_bot()
