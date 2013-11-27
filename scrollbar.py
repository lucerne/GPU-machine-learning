#$Id: scrollbar.py,v 1.3 2004/03/17 04:29:31 mandava Exp $

#scrolled list box is a convinience widget providing a list box widget..

#'items' is a tuple containing initial items to be displayed by the list box
#component.

#'selectioncommand' specifies a function to call when mouse button is
#single clicked over an entry in the listbox component.

# similarly a function called 'dblclickcommand'can be used to specify a
# function to be called when mouse button is double clicked over an entry.

# the size of the megawidget is determined by the width and heigth options
# of the hull component. usehullsize, hull_width and hull_height determine
# the size of the hull component.

#'setlist' method is used to replace all the items of the list box with those
#specified by the items sequence.

#'getcurselection' returns the text of the currently selected items of the listbox.

#'get' method returns all the elements in the list if either first or last
#is not specified.





title = 'Pmw.ScrolledListBox'

# Import Pmw from this directory tree.
import sys
sys.path[:0] = ['../../..']

import Tkinter
import Pmw

class Demo:
    def __init__(self, parent):
	# Create the ScrolledListBox.
	self.box = Pmw.ScrolledListBox(parent,
		items=('Sydney', 'Melbourne', 'Brisbane'),
		labelpos='nw',
		label_text='Cities',
		listbox_height = 6,
		selectioncommand=self.selectionCommand,
		usehullsize = 1,
		hull_width = 200,
		hull_height = 200,
	)

	
	self.box.pack(fill = 'both', expand = 1, padx = 5, pady = 5)

#add some more entries  to the listbox.
	items = ('Andamooka', 'Coober Pedy', 'Innamincka', 'Oodnadatta')
	self.box.setlist(items)
	self.box.insert(2, 'Wagga Wagga', 'Perth', 'London')
	self.box.insert('end', 'Darwin', 'Auckland', 'New York')
	index = list(self.box.get(0, 'end')).index('London')
	self.box.delete(index)
	self.box.delete(7, 8)
	self.box.insert('end', 'Bulli', 'Alice Springs', 'Woy Woy')
	self.box.insert('end', 'Wallumburrawang', 'Willandra Billabong')

        
    def selectionCommand(self):
	sels = self.box.getcurselection()
	if len(sels) == 0:
	    print 'No selection'
	else:
	    print 'Selection:', sels[0]

 
   
######################################################################

# Create demo in root window for testing.
if __name__ == '__main__':
    root = Tkinter.Tk()
    Pmw.initialise(root)
    root.title(title)

    exitButton = Tkinter.Button(root, text = 'Exit', command =
root.destroy)
    exitButton.pack(side = 'bottom')
    widget = Demo(root)
    root.mainloop()


