# DATA 410 - Advanced Machine Learning

### Hello World!

My name is Luke Denoncourt, and I am an excited senior from Windsor, Virginia. I am a biology and data science double major while working on my honors thesis, 
which is utilizing a machine learning model to predict the fitness of a Milkweed plant based upon its soil microbiome. I also am enjoying my third year of working in 
Residence Life here at W&M while impatiently waiting to hear back from graduate school decisions for the Microbiology PhD programs I applied to. On campus, you will 
find me most often:
 - Wandering around outside gawking at the tall trees
 - Listening for owls on the trails while watching the moon 
 - Cooking in my room and trying not to set off the fire alarm. 
 
 At home, our family cat Dyson (like the vacuum) likes to be my sous chef, but, this time, he didn’t enjoy what the Rock was cookin’.

![Dyson living life](https://user-images.githubusercontent.com/67921793/151710233-a57653e7-7f00-4fc5-9802-5c31c1e19bfc.png)

### My favorite math equation is Kepler's third law of planetary motion

![math](http://www.sciweavers.org/tex2img.php?eq=T%5E2%20%3D%20%20%5Cfrac%7B4%2Api%5E2%7D%7BG%28M%2Bm%29%7D%20r%5E3&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

Where **T^2** is the orbital period, 
**G** is the gravitational constant, 
**M** is the mass of the planet, 
**m** is the mass of the moon, 
and **r^3** is the radius of the moon's orbit.

I am working to establish a career in Astrobiology, so detailing the characteristics of planets appeals to my joy of Astronomy. 


### A fun piece of python code an automatic emailer that I use frequently when I have to run code and leave since it takes so long to complete
The email code was used referencing a tutorial at the link: https://realpython.com/python-send-email/

I also used this to make a dad joke emailer that emails a random dad joke whenever the script is ran!

```Python
#Required packages to impliment from command line:
import argparse
import getpass

def dad_joke_emailer(email_input, password_input):

    """
    This python script is to email 1 dad joke that is sent from an email to that same email. Important note: you have
    to turn "less secure app access" on in your email settings to use this email sender. I have set up a junk email
    that I only use for this purpose.
    Arguments:
        email [str] -- your full email
        password [str] -- your password to that email
    """
    ## Required Packages
    import random
    import ssl
    import smtplib
    import os

    jokes = [
        "What kind of noise does a witch's vehicle make? Brooooom Broooom",
        "What's brown and sticky? A stick",
        "Two guys walked into a bar. The third guy ducked",
        "Why are elevator jokes so good? They work on many levels",
        "My wife asked me to go get 6 cans of Sprite from the grocery store. I realized when I go home that I have picked 7 up.",
        "Why do bees have sticky hair? Because they use a honeycomb",
        "Why do some couples go to the gym? Because they want their relationship to work out",
        "How can you tell it's a dogwood tree? By the bark.",
        "My boss told me to have a good day, so I went home.",
        "Why did the man fall down he well? Because he couldn't see that well.",
        "When does a joke become a 'dad joke?' When it becomes apparent.",
        "Why is peter pan always flying? Because he Neverlands",
        "Which state has the most streets? Rhode Island",
        "I used to hate facial hair, but then it grew on me.",
        "Why did the coach go to the bank? To get his quarterback",
        "How do celebrities stay cool? They have many fans",
        "Sundays are always a little sad, but the day before is a sadder day",
        "5/4 people admit that they are bad at fractions",
        "You're American when you go into a bathroom and when you come out, but what are you while you are in the bathroom? European",
        "I've been thinking about taking up meditation. I figure it's better than sitting around and doing nothing",
        "Dogs can't operate MRI machines. But catscan",
        "Singing in the shower is fun until you get soap in your mouth. Then it becomes a soap opera.",
        "It takes guts to be an organ donor.",
        "I lost my job at the bank on my first day. A woman asked me to check her balance, so I pushed her over.",
        "How do you row a canoe filled with puppies? Bring out the doggy paddle.",
        "How does a penguin build his house? Igloos it together",
        "What kind of music do chiropractors like? Hip pop.",
        "What does a house wear? Address",
        "I was going to tell a time-traveling joke, but you guys didn't like it",
        "What does the accountant say while auditing a document? This is taxing.",
        "What do you call a toothless bear? A gummy bear",
        "Why couldn't the bicycle stand up by itself? It was two-tired",
        "What does a nosey pepper do? It gets jalapeno business.",
        "I know a lot of jokes about retired people, but none of them work",
        "What do you call two octopuses that look the same? Itenticle",
        "Sore throats are a pain in the neck"
    ]

    a = random.sample(range(len(jokes)),1)

    jokes_to_send = jokes[a[0]]

    ## Email setup to alert you of model completion and output info
    # For this to work you may need to change your setting 'less secure app access' to ON
    # I made a junk email for this specifically and suggest you do the same
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = email_input  # Enter your address
    receiver_email = email_input  # Enter receiver address

    ## Sending message
    message = """Subject: Hi there! Here is your joke!
    {dad_joke}
    Have a great day! :)
    This message is sent from the dad_joke_emailer.""".format(dad_joke = jokes_to_send)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)
    except:
        print('Did you enter your email and password correctly and did you also change the "LESS SECURE APP ACCESS" setting to ON in your GOOGLE ACCOUNT settings?')


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Emails a dad joke to your email from your email.")

    p.add_argument("-e", "--email", default = 'None', type = str,
                  help=""" Put in your entire email... example@gmail.com. 
                  Make sure to turn 'Less secure app access' ON in your GOOGLE ACCOUNT security settings. """)


    args = p.parse_args()

    print('')
    print('Looks like somebody wants a joke!')
    print('')
    password = getpass.getpass(prompt = 'What is your email password?')

    dad_joke_emailer(args.email, password)
