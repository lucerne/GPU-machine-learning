#$Id: toplevel.py,v 1.2 2004/03/17 04:29:31 mandava Exp $

#We start by importing the Tkinter module, it contains all the
#classes,functions and
#other things needed to work with the Tk tool kit.We can simply import
#everything from Tkinter into our modules namespace.
#                                                                                                   
from Tkinter import *

#To initialize Tkinter, we have to create a Tk root widget which is an
#ordinary window with a title bar.we should only create one root widget for 
#each program and it must be created before any other widgets. 

root= Tk();

#the next option specifies a title to the root or container which is
#displayed on top of the window.

root.title('Toplevel')

#Next we create a label widget as a child to the root window.The label
#widget can display either text or an icon or other image.if it is text a
#text option is used to display which text it is.we then call the pack
#method on this widget which tells it to size itself according to the 
#options mentioned.Before this happens we have to enter the Tkinter event
#loop.

Label(root,text='This is the Toplevel').pack(pady=10)

# Tkinter event loop
root.mainloop()

#The program will stay in the event loop until we close the window.
#The event loop not only handles events from the user such as mouse clicks
#or presses,but also handles operations queued by Tkinter itself such as
#geomentry management queued by pack mathod and display updates.
#The application window will not appear before you enter the main loop.
 

