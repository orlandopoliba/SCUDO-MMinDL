# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.19.6",
# ]
# ///

import marimo

__generated_with = "0.19.6"
app = marimo.App(
    width="medium",
    app_title="Python Classes",
    css_file="style.css",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Python classes

    Python is an object-oriented programming language. *Classes* are used as a blueprint to define new objects. A class defines a set of *attributes* and *methods* that an object of the class will have. An *attribute* is a variable that is associated with the class. A *method* is a function that is associated with the class. Using classes, we can create new *instances* of objects that have the same attributes and methods as the class.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example: the rectangle class

    We define a new class called `Rectangle`. It has the following attributes:

    - `width`: the width of the rectangle
    - `height`: the height of the rectangle

    and the following methods:

    - `get_area()`: returns the area of the rectangle
    - `get_perimeter()`: returns the perimeter of the rectangle
    - `scale(factor)`: scales the rectangle by a given factor

    One method is special when defining a class: the `__init__` method. This method is called when an instance of the class is created. It is used to initialize the attributes of the class.

    The `self` parameter is used to reference to the instance of the class itself.

    As always in programming, it is a good practice to include documentation.
    """)
    return


@app.class_definition
class Rectangle:

    """
    A class to represent a rectangle with a given width and height. One can calculate its area and perimeter, and scale its dimensions.
    """

    def __init__(self, width, height):
        """
        Initializes the Rectangle instance with width and height.

        Args:
        - width (float): The width of the rectangle.
        - height (float): The height of the rectangle.
        """
        self.width = width
        self.height = height

    def get_area(self):
        """
        Calculates the area of the rectangle.

        Returns:
        - float: The area of the rectangle.
        """
        return self.width * self.height

    def get_perimeter(self):
        """
        Calculates the perimeter of the rectangle.

        Returns:
        - float: The perimeter of the rectangle.
        """
        return 2 * (self.width + self.height)

    def scale(self, factor):
        """
        Scales the dimensions of the rectangle by a given factor.

        Args:
        - factor (float): The scaling factor.
        """
        self.width = factor*self.width
        self.height = factor*self.width


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We create an instance of the class by calling the class name as if it were a function, passing the required arguments to the `__init__` method.
    """)
    return


@app.cell
def _():
    rect = Rectangle(2,3)
    return (rect,)


@app.cell
def _(mo):
    mo.md(r"""
    **Important**: different instances of the same class have their own attributes. This becomes clear in the following examples.

    We refer to instances attributes using the dot notation.
    """)
    return


@app.cell
def _():
    other_rect = Rectangle(4,5)
    return (other_rect,)


@app.cell
def _(rect):
    rect.width
    return


@app.cell
def _(other_rect):
    other_rect.width
    return


@app.cell
def _(rect):
    rect.width = 5
    return


@app.cell
def _(other_rect, rect):
    rect.width, other_rect.width
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can call the methods of the class using the dot notation.
    """)
    return


@app.cell
def _(rect):
    rect.width, rect.height
    return


@app.cell
def _(rect):
    rect.get_area()
    return


@app.cell
def _(rect):
    rect.scale(2)
    return


@app.cell
def _(rect):
    rect.get_perimeter()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Subclasses

    Subclasses are classes that inherit attributes and methods from a parent class (referred to as `__super__`). This allows us to create new classes that are based on existing classes, while adding new attributes and methods or modifying existing ones.
    """)
    return


@app.class_definition
class Square(Rectangle):
    
    """
    A class to represent a square, which is a special case of a rectangle where width and height are equal.
    """

    def __init__(self, side_length):
        """
        Initializes the Square instance with side length.

        Args:
        - side_length (float): The length of each side of the square.
        """
        super().__init__(side_length, side_length)


@app.cell
def _():
    square = Square(4)
    return (square,)


@app.cell
def _(square):
    square.get_area()
    return


if __name__ == "__main__":
    app.run()
