#$Id: buttontk.py,v 1.2 2004/03/17 04:29:31 mandava Exp $
#this is a simple program which creates buttons.

import Tkinter
from Tkinter import *
root =Tk()
root.title('Button')
Label(text='I am a button').pack(pady=15)

#button are labels that react to mouse and keyboard events.
#in this case the button is packed to be at the bottom of the root or
#toplevel widget.

Button( text='Button') .pack(side=BOTTOM)
root.mainloop()



