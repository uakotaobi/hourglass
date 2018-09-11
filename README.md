# Hourglass Puzzle Solver
*You have an X-minute hourglass and a Y-minute hourglass.  Use them to measure Z minutes of time.*

## Synopsis

This command-line utility, written in a weekend, solves the (X, Y, Z)
hourglass problem for arbitrary X, Y, and Z using a brute-force decision tree
search.

The decision tree represents all operations that we can perform on two
hourglasses:

| Decision type | Description                                 |
| ------------- | ------------------------------------------- |
| Reset         | Resets both hourglasses                     |
| Flip A        | Flips the first hourglass                   |
| Flip B        | Flips the second hourglass                  |
| Flip Both     | Flips both hourglasses                      |
| Drain Either  | Waits for either hourglass to empty         |
| Drain Both    | Waits for both hourglasses to empty         |
| Start Timing  | Subsequent drains count toward the solution |

*Reset* is the root node of the decision tree, and is only allowed once.
Thereafter, most decisions are allowed as children so long as a handful of
rules are observed:
1. Flips can only follow draining operations;
1. Drains cannot follow other draining operations;
1. You can only start the timer once, and not after a flipping operation;
1. You can only drain both hourglasses if at least one still has sand in it; and
1. You can only drain either hourglass if they _both_ still have sand in them.

Each path from the root of this decision tree down to its leaves then
represents a "simulation" of the hourglass puzzle, with the rules serving to
prune this tree down to a manageable size.  The program runs each simulation
until it either _converges_ on a solution or until it _diverges_ by exceeding
the maximum time or maximum search depth.  Each recursive call returns the
best solution seen so far, and the final solution is printed out as a series
of steps.

Solutions without any setup (i.e., solutions that start the formal timer
immediately) are considered better than solutions that spend time on setup,
and solutions that measure the desired time in fewer steps are better than
solutions that use more steps.

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
    ./hourglass.py 13 11 27
    ```

    This solution requires 11 minutes of setup.

2. Increasing the search depth may allow the program to find a better solution:

    ``` shell
    ./hourglass.py 13 11 27 -d 16
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
