#!/usr/bin/env python
"""Conway's Game of Life in Pygame.

For the hasty:
    Run the following command in your terminal:
        $python -O cgol.py

    Click the close button to close the window.
    Hit Ctrl+C in the command line anytime to exit.

The Game of Life, also known simply as Life, is a cellular automaton
devised by the British mathematician John Horton Conway in 1970.

The "game" is a zero-player game, meaning that its evolution is
determined by its initial state, requiring no further input. One
interacts with the Game of Life by creating an initial configuration
and observing how it evolves.

Ever since its publication, Conway's Game of Life has attracted much
interest, because of the surprising ways in which the patterns can
evolve. Life provides an example of emergence and self-organization.

This is a graphical visualization of Game of Life using Pygame,
so you need to have Pygame installed to run it.
Installation is quick and easy. Download it here:
    http://www.pygame.org/download.shtml

This module makes use of Python 2.7 functionality,
so you will need this version installed to run it.
If you opt for Python 3, Python 3.2 or later is required.

References:
    http://en.wikipedia.org/wiki/Conway's_Game_of_Life
    http://www.pygame.org/

=======================================================================

Command line usage:
  $python -O cgol.py [-h]
                     [-d ROWSxCOLUMNS] [-g GENERATIONS] [-c]
                     [-r WIDTHxHEIGHT] [-f] [-t SPEED] [-n]
                     [-i INFILE] [-o OUTFILE]
                     [-s [FILE]] [-l [FILE]]
                     [-p]
                     [-v] [-V] [-C] [-L]

optional arguments:
  -h, --help            show this help message and exit

Game options:
  -d ROWSxCOLUMNS, --dims ROWSxCOLUMNS
                        grid dimensions (default: 72x128)
  -g GENERATIONS, --generations GENERATIONS
                        number of generations (0: no limit; default: 0)
  -c, --count           count the live cells at the last generation

Display options:
  -r WIDTHxHEIGHT, --resolution WIDTHxHEIGHT
                        window resolution (default: system dependent)
  -f, --fullscreen      run game in fullscreen mode
  -t SPEED, --speed SPEED
                        generation succession speed (FPS; default: 15)
  -n, --nodisplay       do not use a GUI; display options are ignored

Input/Output options:
  -i INFILE, --infile INFILE
                        read the grid from infile
  -o OUTFILE, --outfile OUTFILE
                        write the grid to outfile

Save/Load options:
  -s [FILE], --save [FILE]
                        save game to FILE (default: 'cgol.save')
  -l [FILE], --load [FILE]
                        load game from FILE (default: 'cgol.save')

Profile options:
  -p, --profile         show profiling output

Misc options:
  -v, --verbose         print detailed progress messages
  -V, --version         show program's version number and exit
  -C, --copyright       show copyright message and exit
  -L, --license         show license name and exit

Usage examples:
    $ python -O cgol.py
    $ python -O cgol.py -d 72x128
    $ python -O cgol.py -f -t 30
    $ python -O cgol.py -r 1280x720
    $ python -O cgol.py -g 1000 -t 60
    $ python -O cgol.py -g 100000 -n -c -v
    $ python -O cgol.py -i grid.txt -o newgrid.txt
    $ python -O cgol.py -r 1280x720 -g 1000 -s=game.save
    $ python -O cgol.py -d 72x128 -g 250 -c -r 1280x720 -o=grid.txt

Enjoy!

"""

from __future__ import print_function, division

import itertools
import os
import pprint
import random
import sys
from abc import ABCMeta, abstractmethod
from ast import literal_eval
from collections import namedtuple

# It makes more sense for the user to check the Python version first.
if sys.version_info < (2, 7) or (3, 0) < sys.version_info < (3, 2):
    raise RuntimeError("This module needs Python 2.7 or 3.2+.")

try:
    import pygame
except ImportError:
    raise ImportError("You need to have Pygame installed to run this module.")

__all__ = [
    # Exceptions
    "GameOfLifeError",
    "FileFormatError",
    "DisplayError",

    # ABCs/Interfaces
    "Saveable",
    "Loadable",
    "GameOfLifeObserver",

    # Classes
    "BlindObserver",
    "GameOfLifeWindow",
    "GameOfLife",
]
__author__ = "Vasilis Poulimenos"
__copyright__ = "Copyright (c) 2014, Vasilis Poulimenos"
__license__ = "BSD 3-Clause"
__version__ = "1.0.0"
__date__ = "3/9/2014"

# Fix some slight differences between Python 2 and 3.
if sys.version_info < (3, 0):
    from future_builtins import *
    range = xrange

# Center the game window.
# http://www.pygame.org/wiki/FrequentlyAskedQuestions
os.environ['SDL_VIDEO_CENTERED'] = '1'

pygame.init()


################################## Exceptions ##################################


class GameOfLifeError(Exception):
    """Base class for exceptions in this module.

    It can be used to handle all custom exceptions raised by the module
    (i.e. excluding Python built-in exceptions, such as TypeError).

    """

    def __init__(self, msg):
        Exception.__init__(self, msg)
        self.msg = msg

    def __str__(self):
        return repr(self.msg)


class FileFormatError(GameOfLifeError):
    """Exception raised for errors in input file formats.

    Attributes:
        msg -- explanation of the error

    """
    pass


class DisplayError(GameOfLifeError):
    """Exception raised for errors in the display.

    Attributes:
        msg -- explanation of the error

    """
    pass


############################### ABCs/Interfaces ################################


def _add_metaclass(metaclass):
    """Class decorator for creating a class with a metaclass.

    Taken directly from the Python six project.
    https://bitbucket.org/gutworth/six/src
    http://stackoverflow.com/a/18513858/1751037

    """
    def wrapper(cls):
        orig_vars = cls.__dict__.copy()
        orig_vars.pop('__dict__', None)
        orig_vars.pop('__weakref__', None)
        slots = orig_vars.get('__slots__')
        if slots is not None:
            if isinstance(slots, str):
                slots = [slots]
            for slots_var in slots:
                orig_vars.pop(slots_var)
        return metaclass(cls.__name__, cls.__bases__, orig_vars)
    return wrapper


class _abstractclassmethod(classmethod):
    """A decorator indicating abstract class methods.

    Similar to abstractmethod.

    Taken and adjusted from the python sources.
    http://hg.python.org/cpython/file/3.4/Lib/abc.py
    http://stackoverflow.com/a/11218474/1751037

    """

    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super(_abstractclassmethod, self).__init__(callable)


@_add_metaclass(ABCMeta)
class Saveable(object):
    """Abstract base class for indicating saveable observers.

    Observers that support saving can derive from this ABC.

    """
    @_abstractclassmethod
    def save(observer, file):
        """Save the observer to the file specified.

        Override this method if you want to support saving.

        """
        raise NotImplementedError


@_add_metaclass(ABCMeta)
class Loadable(object):
    """Abstract base class for indicating loadable observers.

    Observers that support loading can derive from this ABC.

    """
    @_abstractclassmethod
    def load(file):
        """Load and return the observer object from the file specified.

        Override this method if you want to support loading.

        """
        raise NotImplementedError


@_add_metaclass(ABCMeta)
class GameOfLifeObserver(object):
    """An observer abstract base class for Game of Life.

    GameOfLife expects an object with this interface.

    """
    @abstractmethod
    def update(self, changes):
        """Perform the changes specified.

        The changes should be provided as a
        list of tuples of the form (i, j, color).

        """
        raise NotImplementedError


del _add_metaclass
del _abstractclassmethod


############################# Game of Life Classes #############################


class BlindObserver(GameOfLifeObserver):
    """An observer that does nothing.

    Useful when no observer is required.

    """

    def __repr__(self):
        return self.__class__.__name__ + "()"

    def update(self, changes):
        pass


class GameOfLifeWindow(GameOfLifeObserver, Saveable, Loadable):
    """The Game of Life window.

    Handles the game graphics and the interaction with the user.

    """

    WHITE = pygame.Color(255, 255, 255)
    BLACK = pygame.Color(0, 0, 0)

    _PYGAME_FLAGS = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF

    _info = pygame.display.Info()
    CURRENT_RES = (_info.current_w, _info.current_h)
    del _info

    _res = pygame.display.list_modes(0, _PYGAME_FLAGS)
    DEFAULT_RES = _res[len(_res)//2]
    del _res


    @classmethod
    def save(cls, observer, file):
        """Save the observer to the file specified.

        The file should be opened in text mode.
        The output format is human-readable.
        
        """
        print(cls.__name__, file=file)
        print(observer.dims, file=file)
        print(observer.resolution, file=file)
        print(observer.fullscreen, file=file)
        print(observer.framerate, file=file)
        print(observer.caption, file=file)
        print(file=file)

    @classmethod
    def load(cls, file):
        """Load and return the observer from the file specified.

        The file should be opened in text mode.

        In case of an invalid file format, FileFormatError is raised.
        
        """
        try:
            dims = literal_eval(next(file))
            resolution = literal_eval(next(file))
            fullscreen = literal_eval(next(file))
            framerate = int(next(file))
            caption = next(file)
        except (ValueError, SyntaxError):
            raise FileFormatError("invalid {} format".format(cls.__name__))
        except StopIteration:
            raise FileFormatError("truncated file")
        return cls(dims, resolution, fullscreen, framerate, caption)


    def __init__(self, dims, resolution=None, fullscreen=False,
                 framerate=15, caption="Conway's Game of Life"):
        """Initialize a new window.

        Arguments:
            dims -- grid width and height
            resolution -- window width and height
                If None, then:
                    it becomes CURRENT_RES if 'fullscreen' is true
                    or DEFAULT_RES, otherwise.
            fullscreen -- run in fullscreen mode
            framerate -- generation succesion rate in FPS.
                If 'framerate' is zero or negative, then the game will
                compute the generations as fast as possible (depended
                on computer speed). If it is too large, then the game
                will not be able to compute the generations that fast
                and will simply run at maximum speed.
            caption -- window title

        """
        # Restart the display for each run to avoid hangs.
        pygame.display.init()

        if resolution is None:
            if fullscreen:
                resolution = GameOfLifeWindow.CURRENT_RES
            else:
                resolution = GameOfLifeWindow.DEFAULT_RES

        self.dims = dims
        self.resolution = resolution
        self.fullscreen = fullscreen
        self.framerate = max(int(framerate), 0)
        self.caption = caption

        flags = GameOfLifeWindow._PYGAME_FLAGS if fullscreen else 0
        try:
            self._screen = pygame.display.set_mode(resolution, flags)
        except pygame.error:
            raise DisplayError("unsupported resolution")
        self._tick = pygame.time.Clock().tick
        
        self._screen.fill(GameOfLifeWindow.WHITE)
        pygame.display.set_caption(caption)

    def __del__(self):
        """ """
        pygame.display.quit()

    def __repr__(self):
        return "{}({}, {}, {}, {}, {})".format(
            self.__class__.__name__, self.dims, self.resolution,
            self.fullscreen, self.framerate, self.caption)

    def update(self, changes):
        """
        
        """
        # Intensive computations ahead, so cache the lookups.
        white, black = GameOfLifeWindow.WHITE, GameOfLifeWindow.BLACK
        # Notice the inversion of indices.
        width = self.resolution[0] // self.dims[1]
        height = self.resolution[1] // self.dims[0]
        fill = self._screen.fill

        rects = []
        for y, x, color in changes:
            rectangle = (x * width, y * height, width, height)
            # white == 0, black == 1
            fill(black if color else white, rectangle)
            rects.append(rectangle)
        pygame.display.update(rects)
        self._tick(self.framerate)


class GameOfLife(object):
    """Conway's Game of Life class.

    The core class that handles the Game of Life logic.

    The universe of the Game of Life is an infinite two-dimensional
    orthogonal grid of square cells, each of which is in one of two
    possible states, alive or dead. Every cell interacts with its
    eight neighbours, which are the cells that are horizontally,
    vertically, or diagonally adjacent.

    Simplest call example (runs a random game infinitely):
        >>> GameOfLife.random().advance()

    This class supports the following built-in functions:
        repr:   Return a valid Python expression suitable for eval.
        str:    Return a human readable string representation.
        len:    Return the total number of cells.

    This class also supports:
        Equality and inequality testing.
            Two games are equal if they have are in the same state
            (useful for comparing games at different generations.)
        Indexing and slicing with the [] operator.
            Negative indexes are supported.
        Iteration (but not reverse iteration as it doesn't make sense).
            There are two iterators provided:
                Calling the iter built-in function: The iterator
                    returned by this function returns one cell
                    at a time in the order found in the grid.
                Calling self.enumerate: The iterator returned yields
                    pairs containing the coordinates of the cell (as
                    a (i, j) tuple) and the value of the cell itself.
                    The iteration order is the same as in iter.
        The sys.getsizeof module function.
            Returns an estimate of the size of the instance (in bytes).
            Mainly for debugging, testing and optimization purposes.

    The following attributes contain extra information:
        observer: the observer for this game
        dims: a tuple with the grid dimensions.
        haschanged: a boolean indicating if there have been
            any changes since the last call to `advance`.

    """

    # Implementation Note: True or 1 is live, False or 0 is dead.

    __slots__ = 'haschanged _grid _observer _neighbors'.split()

    NEIGHBOR_COUNT = 8
    DEFAULT_DIMS = (72, 128)
    DEFAULT_OBSERVER = BlindObserver()

    def _checkrep(self):
        """Invariant checker. Always returns True."""
        grid = self._grid
        assert all(len(row) == len(grid[0]) for row in grid)
        assert all(cell in [0, 1] for row in grid for cell in row)

        neighbors = self._neighbors
        assert sum(1 for row in grid for cell in row) == len(neighbors)
        assert all(len(n) == GameOfLife.NEIGHBOR_COUNT for n in neighbors)
        return True

    def _find_neighbors(self, len=len, range=range):
        """Calculate the coordinates of the neighbors of each cell.

        Used later for fast application of the rules.

        """
        nrows, ncols = len(self._grid), len(self._grid[0])
        self._neighbors = [
            [
                ((i-1) % nrows, (j-1) % ncols),     # Up and Left
                ((i-1) % nrows, j),                 # Up
                ((i-1) % nrows, (j+1) % ncols),     # Up and Right
                (i,             (j-1) % ncols),     # Left
                (i,             (j+1) % ncols),     # Right
                ((i+1) % nrows, (j-1) % ncols),     # Down and Left
                ((i+1) % nrows, j),                 # Down
                ((i+1) % nrows, (j+1) % ncols)      # Down and Right
            ]
            for i in range(nrows) for j in range(ncols)
        ]

    def _init_observer(self, enumerate=enumerate):
        """Tell the initial state of the game to the observer."""
        self._observer.update(
            (i, j, 1)
            for i, row in enumerate(self._grid)
            for j, cell in enumerate(row))

    def _init(self, grid, observer=DEFAULT_OBSERVER):
        """Initialize a Game of Life object.

        This method does not perform any input sanity checks.

        """
        self._grid = grid
        self._observer = observer
        self.haschanged = False
        self._find_neighbors()
        self._init_observer()
        assert self._checkrep()


    @staticmethod
    def savegrid(game, file=None):
        """Save a representation of the game grid to the file specified.

        If file is None, `sys.stdout` is assumed.
        The file should be opened in text mode.

        The grid is represented as a series of lines,
        with each line corresponding to each row.

        """
        if file is None:
            file = sys.stdout
        for row in game._grid:
            print(''.join(str(int(cell)) for cell in row), file=file)

    @staticmethod
    def loadgrid(file=None):
        """Load and return the grid from the file specified.

        If file is None, `sys.stdin` is assumed.
        The file should be opened in text mode.

        The grid is expected to be in the format
        used by the `GameOfLife.savegrid` method.
        Otherwise, FileFormatError is raised.

        """
        if file is None:
            file = sys.stdin
        try:
            grid = [list(map(int, line.rstrip())) for line in file]
        except ValueError:
            raise FileFormatError("invalid grid format")
        else:
            return grid

    @classmethod
    def save(cls, game, file=None):
        """Save the game in the file specified.

        The file should be opened in text mode.
        If file is None, `sys.stdout` is assumed.
        The output format is human-readable.

        """
        if file is None:
            file = sys.stdout
        print("Conway's Game of Life - Version", __version__, file=file)
        print("Number of rows:", len(game._grid), file=file)
        print("Number of cols:", len(game._grid[0]), file=file)
        print(file=file)
        GameOfLife.savegrid(game, file)
        print(file=file)

    @classmethod
    def load(cls, file=None):
        """Load and return the game from the file specified.

        The file should be opened in text mode.
        If file is None, `sys.stdin` is assumed.

        In case of an invalid file format, FileFormatError is raised.

        """
        if file is None:
            file = sys.stdin
        try:
            version = next(file)
            if not version.startswith("Conway's Game of Life - Version "):
                raise FileFormatError("invalid version format")

            try:
                nrows = int(next(file).rsplit(' ', 1)[0])
                ncols = int(next(file).rsplit(' ', 1)[0])
            except ValueError:
                raise FileFormatError("invalid dimension format")

            next(file)          # Skip empty line
            seed = iter(file.readline, '\n')
            grid = GameOfLife.loadgrid(seed)

        except StopIteration:
            raise FileFormatError("truncated file")

        if len(grid) != nrows:
            msg = "expected {} rows, got {}".format(nrows, len(grid))
            raise FileFormatError(msg)
        for i, row in enumerate(grid):
            if len(row) != ncols:
                msg = "expected {} columns in row {}, got {}".format(
                    ncols, i, len(grid))
                raise FileFormatError(msg)

        self = cls.__new__(cls)
        self._init(grid)
        return self


    @property
    def observer(self):
        """The observer for this game."""
        return self._observer

    @observer.setter
    def observer(self, observer):
        self._observer = observer
        self._init_observer()

    @property
    def dims(self):
        """Return a tuple with the grid dimensions ((nrows, ncols))."""
        return (len(self._grid), len(self._grid[0]))


    @classmethod
    def random(cls, dims=DEFAULT_DIMS, observer=DEFAULT_OBSERVER):
        """Return a Game of Life object with a random state.

        Arguments:
            dims -- A 2-tuple containting the number of
                rows and columns.
            observer -- An observer for game of life.

        """
        repeat, choice = itertools.repeat, random.choice
        nrows, ncols = dims

        self = cls.__new__(cls)
        grid = [[choice(choices) for _ in repeat(None, ncols)]
                for choices in repeat([True, False], nrows)]
        self._init(grid, observer)
        return self
    
    @classmethod
    def fromgame(cls, game, observer=DEFAULT_OBSERVER):
        """Copy constructor.
        
        
        """
        self = cls.__new__(cls)
        self._init([list(map(int, row)) for row in game._grid], observer)
        return self

    @staticmethod
    def _checkgrid(grid, any=any, len=len):
        rowsize = len(grid[0])
        if any(len(row) != rowsize for row in grid):
            raise ValueError("variable row size")
    
    def __init__(self, seed, observer=DEFAULT_OBSERVER, copy=True):
        """Initialize a Game of Life object.

        Arguments:
            seed -- An iterable of iterables.
                The containing iterables must yield the same number
                of values and each value must evaluate to a boolean.
            observer -- An observer for game of life.
            copy -- A boolean indicating whether to copy the seed.

        Raises ValueError if the seed does not represent a valid grid.

        """
        if copy:
            seed = [list(map(int, row)) for row in seed]
        GameOfLife._checkgrid(seed)
        self._grid = seed
        self._init(self._grid, observer)

    def __repr__(self):
        # Note that the copy argument is set to False.
        return '{}({}, {}, {})'.format(
            self.__class__.__name__, self._grid, self._observer, False)

    def __str__(self):
        # Create a string of the arguments as a tuple.
        # Note that the copy argument is set to False.
        args = pprint.pformat((
            [list(map(int, row)) for row in self._grid],
            self._observer,
            False))
        # Prepend the class name.
        return self.__class__.__name__ + args

    def __eq__(self, other):
        # Two games are the same if their grids are the same.
        if not isinstance(other, GameOfLife):
            return NotImplemented
        return self._grid == other._grid

    def __ne__(self, other):
        return not (self == other)

    def __len__(self):
        return len(self._grid) * len(self._grid[0])

    def __getitem__(self, pos):
        """Return the value of the cell in position pos=(row, col).

        Negative coordinates are interpreted as in list indexing.
        
        Example: game[0, 0]

        """
        row, col = pos
        return self._grid[row][col]

    def __iter__(self):
        """Return an iterator that returns one cell at a time."""
        return itertools.chain.from_iterable(self._grid)

    def __reversed__(self):
        # Reverse iteration does not make much sense here.
        # Raise to stop the interpreter from calling __getitem__.
        raise NotImplementedError

    def __sizeof__(self):
        """Return an estimate of the size of the instance (in bytes)."""
        sizeof = sys.getsizeof
        size = sizeof(self.__slots__)
        size += sum(sizeof(getattr(self, name)) for name in __slots__)
        size += sum(sizeof(row) for row in self._grid)
        size += sum(sizeof(n) for n in self._neighbors)
        return size


    def _advance(self, reps, enumerate=enumerate, divmod=divmod, bool=bool,
                 zip=zip, chain_from_iterable=itertools.chain.from_iterable):
        """Optimized utility method used to run the game efficiently."""
        # Intensive computations ahead, so cache the lookups.
        grid = self._grid
        neighbors = self._neighbors
        update = self._observer.update
        ncols = len(grid[0])

        self.haschanged = False
        for _ in reps:
            # Create the next generation.
            changes = []
            for pos, (cell, neighbors_) in enumerate(zip(
                    chain_from_iterable(grid), neighbors)):
                ((upl1, upl2), (up1, up2), (upr1, upr2),
                 (left1, left2), (right1, right2),
                 (dl1, dl2), (down1, down2), (dr1, dr2)) = neighbors_
                count = (grid[upl1][upl2] + grid[up1][up2] + grid[upr1][upr2] +
                         grid[left1][left2] + grid[right1][right2] +
                         grid[dl1][dl2] + grid[down1][down2] + grid[dr1][dr2])

                if cell:        # black
                    if count < 2 or count > 3:
                        i, j = divmod(pos, ncols)
                        changes.append((i, j, 0))
                else:           # white
                    if count == 3:
                        i, j = divmod(pos, ncols)
                        changes.append((i, j, 1))

            # Make the changes.
            for (i, j, _) in changes:
                grid[i][j] = not grid[i][j]
            update(changes)
            self.haschanged |= bool(changes)
            assert self._checkrep()

    def advance(self, generations=None):
        """Advance the game for the generations specified.

        If `generations` is None, the game will only stop with user input.

        """
        if generations is None:
            self._advance(itertools.repeat(None))
        else:
            self._advance(itertools.repeat(None, generations))

    def all(self, coordit):
        """Return True if all cells specified are live.

        `coordit` must be an iterable that returns
        coordinates as tuples of the form (row, col).
        If the iterable is empty, return True.

        """
        grid = self._grid
        return all(grid[i][j] for i, j in coordit)

    def any(self, coordit):
        """Return True if any cells specified are live.

        `coordit` must be an iterable that returns
        coordinates as tuples of the form (row, col).
        If the iterable is empty, return False.

        """
        grid = self._grid
        return any(grid[i][j] for i, j in coordit)

    def enumerate(self):
        """Return an iterator that yields ((i, j), cell) pairs.

        The pair consists of the coordinates of the cell and the value
        of the cell itself. The iteration order is the same as in iter.

        """
        for i, row in enumerate(self._grid):
            for j, cell in enumerate(row):
                yield ((i, j), cell)

    def countlive(self):
        """Return the number of live cells."""
        return sum(itertools.chain.from_iterable(self._grid))

    def countdead(self):
        """Return the number of dead cells."""
        return len(self) - self.count_live()

    def haslive(self):
        """Return True if there are any live cells."""
        return any(itertools.chain.from_iterable(self._grid))


def golparser():
    """Return a new argument parser for this module.

    Argument handling via the magic of the argparse module.

    """
    import argparse
    import textwrap

    def dims(string):
        """Convert a dimensions string to a (width, height) tuple."""
        try:
            width, height = string.split("x")
        except ValueError:
            msg = "invalid dimensions format: {}".format(string)
            raise argparse.ArgumentTypeError(msg)
        try:
            width = int(width)
        except ValueError:
            msg = "invalid literal for width: {}".format(width)
            raise argparse.ArgumentTypeError(msg)
        try:
            height = int(height)
        except ValueError:
            msg = "invalid literal for height: {}".format(height)
            raise argparse.ArgumentTypeError(msg)

        if width < 0 or height < 0:
            msg = "negative width or height: {}, {}".format(width, height)
            raise argparse.ArgumentTypeError(msg)

        Dims = namedtuple('Dims', ['width', 'height'])
        return Dims(width, height)

    epilog = textwrap.dedent('''\
        Usage examples:
            $ python -O cgol.py
            $ python -O cgol.py -d 72x128
            $ python -O cgol.py -f -t 30
            $ python -O cgol.py -r 1280x720
            $ python -O cgol.py -g 1000 -t 60
            $ python -O cgol.py -g 100000 -n -c -v
            $ python -O cgol.py -i grid.txt -o newgrid.txt
            $ python -O cgol.py -r 1280x720 -g 1000 -s=game.save
            $ python -O cgol.py -d 72x128 -g 250 -c -r 1280x720 -o=grid.txt

        Enjoy!''')
    description = textwrap.dedent('''\
        Conway's Game of Life in Pygame

        For the hasty:
            Run the following command in your terminal:
                $python -O cgol.py

            Click the close button to close the window.
            Hit Ctrl+C in the command line anytime to exit.''')

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=description, epilog=epilog)

    group = parser.add_argument_group("Game options")
    group.add_argument(
        "-d", "--dims", type=dims, metavar="ROWSxCOLUMNS",
        default="{}x{}".format(*GameOfLife.DEFAULT_DIMS),
        help="grid dimensions (default: %(default)s)")
    group.add_argument(
        "-g", "--generations", type=int, default=0,
        help="number of generations (0: no limit; default: %(default)s)")
    group.add_argument(
        "-c", "--count", action="store_true",
        help="count the live cells at the last generation")

    group = parser.add_argument_group("Display options")
    group.add_argument(
        "-r", "--resolution", type=dims, metavar="WIDTHxHEIGHT",
        default="{}x{}".format(*GameOfLifeWindow.DEFAULT_RES),
        help="window resolution (default: %(default)s)")
    group.add_argument(
        "-f", "--fullscreen", action="store_true",
        help="run game in fullscreen mode")
    group.add_argument(
        "-t", "--speed", type=float, default=15,
        help="generation succession speed (FPS; default: %(default)s)")
    group.add_argument(
        "-n", "--nodisplay", action="store_true",
        help="do not use a GUI; display options are ignored")

    group = parser.add_argument_group("Input/Output options")
    group.add_argument(
        "-i", "--infile", type=argparse.FileType('r'),
        help="read the grid from %(dest)s")
    group.add_argument(
        "-o", "--outfile", type=argparse.FileType('w'),
        help="write the grid to %(dest)s")

    group = parser.add_argument_group("Save/Load options")
    group.add_argument(
        "-s", "--save", nargs='?', type=argparse.FileType('w'),
        metavar="FILE", const="cgol.save",
        help="save game to %(metavar)s (default: '%(const)s')")
    group.add_argument(
        "-l", "--load", nargs='?', type=argparse.FileType('r'),
        metavar="FILE", const="cgol.save",
        help="load game from %(metavar)s (default: '%(const)s')")
    # TODO: Add display save/load options

    group = parser.add_argument_group("Profile options")
    group.add_argument("-p", "--profile", action="store_true",
                       help="show profiling output")

    group = parser.add_argument_group("Misc options")
    group.add_argument("-v", "--verbose", action="store_true",
                       help="print detailed progress messages")
    group.add_argument("-V", "--version", action="version",
                       version="Conway's Game of Life " + __version__)
    group.add_argument("-C", "--copyright", action="store_true",
                       help="show copyright message and exit")
    group.add_argument("-L", "--license", action="store_true",
                       help="show license name and exit")
    return parser


def main(args=None):
    """Run the script with args as arguments.

    If `args` is None, then sys.argv is used.
    This function can be used for testing, for quick-and-dirty use
    of the module in another script or while experimenting in the
    interactive prompt.

    Example:
        >>> main('-g 2000 -t 10'.split())

    """

    try:
        args = golparser().parse_args(args)

        if args.verbose:
            print("Conway's Game of Life " + __version__)
            print("Use '-h' (without quotes) in the command line for help.")
            print("Hit Ctrl+C anytime to exit.")

        if args.copyright:
            print(__copyright__)
            return 0
        if args.license:
            print(__license__)
            return 0

        if args.profile:
            # Profiling is going to be extremely rarely used,
            # so import the necessary modules only here.
            try:
                import cProfile as profile
            except ImportError:
                import profile
            import pstats
            # Module (c)StringIO was removed from Python 3.
            try:
                from cStringIO import StringIO
            except ImportError:
                try:
                    from StringIO import StringIO
                except ImportError:
                    from io import StringIO

            pr = profile.Profile()
            pr.enable()

        try:
            if args.nodisplay:
                observer = GameOfLife.DEFAULT_OBSERVER
            else:
                observer = GameOfLifeWindow(
                    args.dims, args.resolution, args.fullscreen, args.speed)
                if args.verbose:
                    print("Observer created.")

            if args.load:
                game = GameOfLife.load(args.load)
                game.observer = observer
                if args.verbose:
                    print("Game loaded.")
                args.dims = game.dims           # Set args.dims manually.
                if args.verbose:
                    print("Game resumed.")
            elif args.infile:
                grid = GameOfLife.loadgrid(args.infile)
                if args.verbose:
                    print("Grid loaded.")
                game = GameOfLife(grid, observer, False)
                args.dims = game.dims           # Set args.dims manually.
                if args.verbose:
                    print("Game started.")
            else:
                game = GameOfLife.random(args.dims, observer)
                if args.verbose:
                    print("Random game created.")
                    print("Game started.")

            game.advance(args.generations or None)      # 0: no limit
            if args.verbose:
                print("Game stopped.")

            if args.count:
                if args.verbose:
                    print("Live cell count:", end=' ')
                print(game.countlive())
            if args.outfile:
                GameOfLife.savegrid(game, args.outfile)
                if args.verbose:
                    print("Grid saved.")
            if args.save:
                GameOfLife.save(game, args.save)
                if args.verbose:
                    print("Game saved.")
        except GameOfLifeError as e:
            print(type(e).__name__ + ":", e.msg, file=sys.stderr)
            return 1
        except Exception as e:
            print(type(e).__name__ + ":", e, file=sys.stderr)
            print("That's all we know. Sorry about that.", file=sys.stderr)
            # Unix programs generally use code 2 for command
            # line syntax errors and argparse does the same.
            return 3

        if args.profile:
            pr.disable()
            s = StringIO()
            ps = pstats.Stats(pr, stream=s).strip_dirs()
            ps.sort_stats('time', 'cumtime').print_stats()
            print(s.getvalue())
            s.close()
    except KeyboardInterrupt:
        if args.verbose:
            print("Game stopped by user.")
        return 0

if __name__ == "__main__":
    sys.exit(main())

# That's all, folks.
