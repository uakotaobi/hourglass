# Hourglass Puzzle Solver
*You have an X-minute hourglass and a Y-minute hourglass.  Use them to measure Z minutes of time.*

## Synopsis

This command-line utility, written in a weekend, solves the (X, Y, Z)
hourglass problem for arbitrary X, Y, and Z using a brute-force decision tree
search.  It uses a handful of intelligent pruning operations to reduce the size
of the tree to a manageable level, and its output is a series of steps.

The program will try to return the best solution it can find.  Solutions
without any setup are better than solutions that spend time on setup, and
solutions that measure the time in fewer steps are better than solutions that
use more steps.

## Usage

```
usage: hourglass.py [-h] [-d DEPTH] FIRST SECOND DESIRED

positional arguments:
  FIRST
                        The number of minutes in the first hourglass.  This
                        must be a positive integer.
  SECOND
                        The number of minutes in the second hourglass.  This
                        must be a positive integer.
  DESIRED
                        The number of minutes to measure using the two
                        hourglasses.  This must be a positive integer.

optional arguments:
  -h, --help            show this help message and exit
  -d DEPTH, --depth DEPTH

                        The maximum search depth when traversing the decision
                        tree (the default is 12.)  If the program is having
                        trouble finding a solution and you know there is one,
                        you may need to increase this.

                        Keep in mind that increasing the maximum depth of the
                        search space increases this program's running time,
                        even if the solution that is found turns out to be
                        trivial.  Also remember that starting the formal timer
                        is an extra step in itself.
```

## Examples

1. Measuring 27 minutes with a 13-minute hourglass and an 11-minute hourglass:

    ``` shell
    hourglass.py 13 11 27
    ```

    This solution requires 11 minutes of setup.

2. Increasing the search depth may allow the program to find a better solution:

    ``` shell
    hourglass.py 13 11 27 -d 16
    ```

    This solution requires no setup and takes exactly 27 minutes, making it a
    "perfect" solution.  Perfect solutions are preferred when they can be
    found.

## Caveats

This program cannot predict in advance, except in the most trivial cases,
whether no solution exists or whether searching at a greater depth might lead
to a better solution.  So if you receive a dissatisfying result, increasing
the search depth *may* yield better results at the cost of CPU usage and
execution time.

## License

This program is released under the GNU General Public License, version 3 or
later.
