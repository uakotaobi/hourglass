#!/usr/bin/env python3
# HOURGLASS.PY -- A brute-force hourglass puzzle solver
# Copyright (C) 2018 Uche Akotaobi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from abc import ABCMeta, abstractmethod
import sys
import argparse
from math import ceil, gcd
from textwrap import dedent


class State:
    """
    Stores the state of an 'hourglass simulation.'  This is moved through
    branches of TreeNodes to simulate what a human being would do with the two
    hourglasses.
    """

    def __init__(self, minutesInFirstHourglass, minutesInSecondHourglass,
                 desiredTime):
        self.aMax = minutesInFirstHourglass
        self.bMax = minutesInSecondHourglass
        self.d = desiredTime
        self.reset()

    def clone(self):
        """Returns a copy of the current state."""
        newState = State(self.aMax, self.bMax, self.d)
        newState.a = self.a
        newState.b = self.b
        newState.f = self.f
        newState.t = self.t
        return newState

    def __str__(self):
        """
        Converts this object into a printable string for debugging purposes.
        """
        return "a={0}, b={1}, f={2}, t={3}, aMax={4}, bMax={5}, d={6}".format(self.a, self.b, self.f, self.t, self.aMax, self.bMax, self.d)

    def reset(self):
        self.a = self.aMax
        self.b = self.bMax
        self.f = -1  # Formal timer; only starts after being set to 0
        self.t = 0   # Total elapsed time

    def getDesiredTime(self):
        """
        Returns the time that we're trying to target with our two hourglasses.
        """
        return self.d

    def getFormalTime(self):
        """
        Returns the total time that has elapsed since the start of formal
        timekeeping.  This will be -1 if formal time has not yet started.
        """
        return self.f

    def getTotalTime(self):
        """
        Returns the total time that has elapsed since the simulation started.
        """
        return self.t

    def getCapacity(self, hourglass):
        """
        Gets the total amount of sand that is held by the given hourglass when
        it is reset (either 'A' or 'B').

        """
        if hourglass.lower() == "a":
            return self.aMax
        elif hourglass.lower() == "b":
            return self.bMax
        else:
            raise Exception("getCapacity: Invalid hourglass '{0}'".format(hourglass))

    def getRemainingTime(self, hourglass):
        """
        Gets the amount of sand that is currently held on the top half of the
        given hourglass (either 'A' or 'B').
        """
        if hourglass.lower() == "a":
            return self.a
        elif hourglass.lower() == "b":
            return self.b
        else:
            raise Exception("getCapacity: Invalid hourglass '{0}'".format(hourglass))

    def flip(self, hourglass):
        """
        Flips the given hourglass (either 'A', 'B', or 'AB' to flip both
        hourglasses.)

        Duration of operation: Instantaneous.
        Side effect: The given hourglass's sand amount is inverted.
        """

        flipped = False
        h = hourglass.lower().strip()

        if h == "a" or h == "ab":
            self.a = self.aMax - self.a
            flipped = True

        if h == "b" or h == "ab":
            self.b = self.bMax - self.b
            flipped = True

        if not flipped:
            raise Exception("flipHourglass: Invalid hourglass '{0}'".format(hourglass))

    def startFormalTimer(self):
        """
        Setup for the hourglasses is complete!  You are ready to start the
        formal timer.

        Duration of operation: Instantaneous.
        Side effect: After calling this function, any attempt to call the
                     drain functions will increment the formal timer as well
                     as the total timer.
        """
        if self.f < 0:
            self.f = 0

    def drainAll(self):
        """Lets both hourglasses drain until both are empty.

        Duration of operation: MAX(A, B), where A and B are the amounts of
                               time in the two hourglasses.

        Side effect: The overall timer and the formal timer are incremented by
                     MAX(A, B).  Both hourglasses will contain 0 minutes of
                     sand on top.
        """
        maxTime = max(self.a, self.b)
        self.t += maxTime
        if self.f >= 0:
            self.f += maxTime
        self.a = 0
        self.b = 0

    def drainEither(self):
        """
        Lets both hourglasses drain until either one is empty.

        Duration of operation: MIN(A, B), where A and B are the amounts of
                               time in the two hourglasses.

        Side effect: The overall timer and the formal timer are incremented by
                     MIN(A, B).  Both hourglasses will be decremented by
                     MIN(A, B), and therefore at least one of the two will be
                     empty.
        """
        minTime = min(self.a, self.b)
        self.t += minTime
        if self.f >= 0:
            self.f += minTime
        self.a -= minTime
        self.b -= minTime


class TreeNode(metaclass=ABCMeta):
    def __init__(self, parentTreeNode):
        """Creates a new TreeNode that is a child of the given one.  Note that this is
        the only way to add children to the tree."""
        self.children_ = []
        self.parent_ = parentTreeNode
        self.depth_ = 0 if not parentTreeNode else parentTreeNode.depth() + 1

    def depth(self):
        """
        Returns the depth of this TreeNode (that is, how much walking you'd
        need to do, starting from the root node, to get to this one.)
        """
        return self.depth_

    def parent(self):
        """
        Returns the parent of this TreeNode, or None if this is a root
        node.
        """
        return self.parent_

    def isRoot(self): return self.parent_ is None

    def children(self):
        """
        Returns an array of all the children of this node.  If the array is
        empty, this node is a leaf node.
        """
        return [childNode for childNode in self.children_]

    def isLeaf(self): return len(self.children_) == 0

    def removeChild(self, childTreeNode):
        """Tries to remove the given child node from this subtree.  Returns True on
        success and False if this node had no such child.

        The function does not attempt to descend the children recursively, so
        it can only remove a at most single immediate child.
        """
        for i in range(len(self.children_)):
            if id(self.children_[i]) == id(childTreeNode):
                del self.children_[i]
                return True
        return False

    # Derived classes must implement these methods.
    @abstractmethod
    def advance(self, state):
        """Advances the game state object by performing this node's operations upon it.

        Returns a human-readable series of sentences can be printed out to
        explain what the node is doing.
        """
        raise Exception("The advance() function must be defined by derived TreeNode classes.")

    @abstractmethod
    def name(self):
        """Returns a brief name for this node, used for debugging."""
        raise Exception("The name() function must be defined by derived TreeNode classes.")


class Reset(TreeNode):
    def name(self): return "RESET"

    def advance(self, state):
        state.reset()
        return "Reset both hourglasses."


class Flip(TreeNode):
    """
    This only exists to make TreeNodes that contain flips easier to identify.
    """
    pass


class FlipA(Flip):
    def name(self): return "FLIP A"

    def advance(self, state):
        state.flip("A")
        qualifier = "" if state.getCapacity("A") != state.getCapacity("B") else "first "
        return "Flip the {0}{1}-minute hourglass.".format(qualifier,
                                                          state.getCapacity("A"))


class FlipB(Flip):
    def name(self): return "FLIP B"

    def advance(self, state):
        state.flip("B")
        qualifier = "" if state.getCapacity("A") != state.getCapacity("B") else "second "
        return "Flip the {0}{1}-minute hourglass.".format(qualifier,
                                                          state.getCapacity("B"))


class FlipBoth(Flip):
    def name(self): return "FLIP BOTH"

    def advance(self, state):
        state.flip("AB")
        return "Flip both hourglasses."


class StartFormalTimer(TreeNode):
    def name(self): return "START FORMAL TIME"

    def advance(self, state):
        state.startFormalTimer()
        return "*Formal time begins.*"


class Drain(TreeNode):
    """
    This only exists to make TreeNodes that contain drains easier to identify.
    """
    pass


class DrainBoth(Drain):
    def name(self): return "DRAIN BOTH"

    def advance(self, state):

        aMax = state.getCapacity("A")
        bMax = state.getCapacity("B")
        aOld = state.getRemainingTime("A")
        bOld = state.getRemainingTime("B")
        drainTime = max(aOld, bOld)
        if drainTime == 0:
            # Precondition violated!
            return "--Internal error-- Can't drain both because both hourglasses are already empty."

        state.drainAll()

        plural = "" if drainTime == 1 else "s"
        message = "Let both hourglasses drain out.  After {0} minute{1}, they will both be empty (the {2}-minute hourglass will empty first.)"
        return message.format(drainTime,
                              plural,
                              aMax if aMax < bMax else bMax)


class DrainEither(Drain):
    def name(self): return "DRAIN EITHER"

    def advance(self, state):

        aMax = state.getCapacity("A")
        bMax = state.getCapacity("B")
        aOld = state.getRemainingTime("A")
        bOld = state.getRemainingTime("B")
        drainTime = min(aOld, bOld)
        if drainTime == 0:
            # Precondition violated!
            return "--Internal error-- Can't drain either because one of the hourglasses is already empty."

        # print("DrainEither before: {0}".format(state))
        state.drainEither()
        # print("DrainEither after: {0}".format(state))

        # Which hourglass are we trying to drain?  You can tell by looking any
        # which one currently has 0 sand on the top half.
        message = ""
        aNew = state.getRemainingTime("A")
        bNew = state.getRemainingTime("B")
        if aNew == 0 and bNew == 0:

            plural = "" if drainTime == 1 else "s"
            message = "Let both hourglasses drain out.  After {0} minute{1}, they will both be empty,"
            message = message.format(drainTime, plural)

        elif aNew > 0:  # and bNew == 0

            drainTimePlural = "" if drainTime == 1 else "s"
            remainingTimePlural = "" if aNew == 1 else "s"
            message = "Let the {0}-minute hourglass drain out.  After {1} minute{2}, the {3}-minute hourglass will have {4} minute{5} left."
            message = message.format(bMax,
                                     drainTime,
                                     drainTimePlural,
                                     aMax,
                                     aNew,
                                     remainingTimePlural)

        else:  # bNew > 0

            drainTimePlural = "" if drainTime == 1 else "s"
            remainingTimePlural = "" if bNew == 1 else "s"
            message = "Let the {0}-minute hourglass drain out.  After {1} minute{2}, the {3}-minute hourglass will have {4} minute{5} left."
            message = message.format(aMax,
                                     drainTime,
                                     drainTimePlural,
                                     bMax,
                                     bNew,
                                     remainingTimePlural)

        return message


def getChain(treeNode):
    """
    This utility function gets the list of TreeNodes going from the given
    node all the way back to the root.

    Note that this is the reverse of the direction we'd use to actually
    traverse the decision tree.
    """
    nodeChain = []
    currentNode = treeNode
    while not currentNode.isRoot():
        nodeChain.append(currentNode)
        currentNode = currentNode.parent()
    nodeChain.append(currentNode)  # Don't forget the root.
    return nodeChain

# TODO: Write a function that optimizes the output of getChain().
# Essentially, if a given hourglass stops being allowed to drain out, then it
# is no longer contributing to the overall solution and is just being
# "flipped/drained along for the ride."


def buildTree(treeNode, state, maxDepth):
    """This recursive function builds a decision tree, prunes the branches, and
    searches for the path through the tree representing the best solution to
    the hourglass problem.

    Returns a tuple.  The first element of the tuple is an array of strings
    representing the best solution that could be found, or an empty array if
    no solution could be found.  The second element of the tuple is the amount
    of total time that elapsed for this solution (or 0 in the case of no
    solution.)
    """
    # Let this node do what it needs to do.  (We'll gather the message string
    # it returns later.)
    treeNode.advance(state)

    # Does this node exceed our time bounds?
    if state.getFormalTime() > state.getDesiredTime():
        return [], 0

    # Can we descend no further?
    if treeNode.depth() > maxDepth:
        return [], 0

    # So we not have enough depth to get to our time goal before we'd run
    # afoul of the depth limit?
    biggestHourglass = max(state.getCapacity("A"), state.getCapacity("B"))
    minimumDrains = ceil((state.getDesiredTime() - state.getFormalTime()) / biggestHourglass)
    # It gets worse than that, since any draining beyond biggestHourglass
    # minutes also needs a flip, too, to reset on or both hourglasses!
    minimumSteps = 2 * minimumDrains - 1
    if state.getFormalTime() < 0:
        # It takes one whole step just to start the formal timer!
        minimumSteps += 1
    if treeNode.depth() + minimumSteps > maxDepth:
        return [], 0

    # Is this node a solution?
    if state.getFormalTime() == state.getDesiredTime():
        # It is!
        nodeChain = getChain(treeNode)

        # We now have a reverse chain of parents, starting with us and
        # traveling all the way up to the root.  Simulate the game again to
        # gather the messages.
        messages = []
        tempState = state.clone()
        tempState.reset()  # Need a reset state here, not a copy.
        for currentNode in reversed(nodeChain):
            messages.append(currentNode.advance(tempState))
        # print()

        formalTimePlural = "minute has" if tempState.getFormalTime() == 1 else "minutes have"
        totalTimePlural = "minute of time has" if tempState.getTotalTime() == 1 else "minutes of time have"
        finalMessage = "*{0} {1} passed* since the formal keeping of time.  {2} {3} passed in total."
        messages.append(finalMessage.format(tempState.getFormalTime(),
                                            formalTimePlural,
                                            tempState.getTotalTime(),
                                            totalTimePlural))

        return messages, tempState.getTotalTime()

    # Time to try new decisions.
    decisions = []
    if state.getRemainingTime("A") > 0 and state.getRemainingTime("B") > 0:
        # Draining either is only possible if we haven't already drained
        # both.
        decisions.append(DrainEither(treeNode))

    # This correctly adds "drain both" as a general operation to the decision
    # tree, but it just massively increases the search space without a lot of
    # seeming benefit.
    #
    if state.getRemainingTime("A") > 0 or state.getRemainingTime("B") > 0:
        # Draining both hourglasses is possible only if at least one of them
        # is non-empty AND we're not already following a DrainEither (why
        # would we drainEither() and then drainAll() when we could just
        # drainAll() in the first place?)
        if not isinstance(treeNode, DrainEither):
            # You can't follow a DrainEither with a DrainBoth--it would be
            # better just to have the DrainBoth by itself in those
            # circumstances.
            decisions.append(DrainBoth(treeNode))

    # Try to start formal time if that hasn't been done yet.
    if state.getFormalTime() < 0 and not isinstance(treeNode, Flip):
        # Formal time should never begin after a flip (we can do the flip
        # after starting formal time.)
        decisions.append(StartFormalTimer(treeNode))

    # We don't want two the follow each other in the decision chain.
    # - FLIP A -> FLIP B is the same as FLIP BOTH.
    # - FLIP B -> FLIP A is the same as FLIP BOTH.
    # - After doing FLIP BOTH, it makes no sense to immediately do another
    #   flip.
    # Also, don't flip right after the reset--that's just silly.
    if not isinstance(treeNode, Flip) and not isinstance(treeNode, Reset):
        # Always try flipping both hourglasses first.
        decisions.append(FlipBoth(treeNode))

        # Drain Both should not be followed by Flip A or Flip B -- both
        # hourglasses are empty, so both should be flipped if we're going to
        # flip at all.
        mostRecentDrain = None
        currentNode = treeNode
        while not currentNode.isRoot():
            if isinstance(currentNode, Drain):
                mostRecentDrain = currentNode
                break
            currentNode = currentNode.parent()
        if not isinstance(mostRecentDrain, DrainBoth):
            decisions.append(FlipA(treeNode))
            decisions.append(FlipB(treeNode))

    # Descend recursively down the tree looking for solutions.
    solutions = []
    for node in decisions:
        messages, totalTime = buildTree(node, state.clone(), maxDepth)
        if len(messages) > 0:
            solutions.append((messages, totalTime))

    if (len(solutions) > 0):
        # Return the best solution (that is, the one that has the smallest
        # total time.)
        #
        # TODO: Find a more Pythonic way to do this.
        bestSolution = None
        bestSolutionTime = 0
        for messages, totalTime in solutions:
            # If there's no solution yet, this is the best one by definition.
            #
            # If there is a best solution but this one takes less totalTime,
            # then this is the best one.
            #
            # If there is a best solution and it takes the same amount of
            # totalTime, but this one has fewer steps, then this one is the
            # best one.
            if not bestSolution or \
               totalTime < bestSolutionTime or \
               (totalTime == bestSolutionTime and len(messages) < len(bestSolution[0])):
                bestSolution = messages, totalTime
                bestSolutionTime = totalTime
        return bestSolution
    else:
        # No solution found within the desired depth.
        return [], 0


# TODO: How do we indicate draining back?

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=dedent("""
    Given an X-minute hourglass and a Y-minute hourglass, this program
    attempts to measure exactly Z minutes of time.  While this program isn't
    perfect, it will find an optimal solution most of the time when one
    exists.

    You'll get more solutions if the two hourglasses are relatively prime.
    """), formatter_class=argparse.RawTextHelpFormatter)

    defaultDepth = 12

    def positive_int(string):
        try:
            n = int(string)
            if n <= 0:
                message = "{0} is not a positive integer"
                raise argparse.ArgumentTypeError(message.format(n))
            return n
        except ValueError as e:
            message = "\"{0}\" is not an integer"
            raise argparse.ArgumentTypeError(message.format(string)) from e

    parser.add_argument("minutesInFirstHourglass",
                        type=positive_int,
                        metavar="FIRST",
                        help=dedent("""
                        The number of minutes in the first hourglass.  This
                        must be a positive integer."""))

    parser.add_argument("minutesInSecondHourglass",
                        type=positive_int,
                        metavar="SECOND",
                        help=dedent("""
                        The number of minutes in the second hourglass.  This
                        must be a positive integer."""))

    parser.add_argument("desiredTime",
                        type=positive_int,
                        metavar="DESIRED",
                        help=dedent("""
                        The number of minutes to measure using the two
                        hourglasses.  This must be a positive integer."""))

    parser.add_argument("-d",
                        "--depth",
                        type=positive_int,
                        required=False,
                        metavar="DEPTH",
                        default=defaultDepth,
                        help=dedent("""
                        The maximum search depth when traversing the decision
                        tree (the default is {0}.)  If the program is having
                        trouble finding a solution and you know there is one,
                        you may need to increase this.

                        Keep in mind that increasing the maximum depth of the
                        search space increases this program's running time,
                        even if the solution that is found turns out to be
                        trivial.  Also remember that starting the formal timer
                        is an extra step in itself.""".format(defaultDepth)))

    args = parser.parse_args()
    minutesInFirstHourglass = args.minutesInFirstHourglass
    minutesInSecondHourglass = args.minutesInSecondHourglass
    desiredTime = args.desiredTime
    maxDepth = args.depth

    state = State(minutesInFirstHourglass,
                  minutesInSecondHourglass,
                  desiredTime)

    factor = gcd(minutesInFirstHourglass, minutesInSecondHourglass)
    if (factor > 1 and desiredTime % factor > 0):
        print(dedent("""
        Warning: The hourglasses can only measure multiples of {0} since that is their greatest common factor.
        """.format(factor)))

    messages, totalTime = buildTree(Reset(None), state, maxDepth)
    if len(messages) == 0:
        print("No solution found at depth {0}.  Try a higher depth?".format(maxDepth))
        sys.exit(1)
    else:
        print()
        perfect = "Found a perfect" if totalTime == state.getDesiredTime() else "Best"
        plural = "" if state.getDesiredTime() == 1 else "s"
        message = "{0} solution for measuring {1} minute{2} with a {3}-minute hourglass and a {4}-minute hourglass:"
        print(message.format(perfect,
                             state.getDesiredTime(),
                             plural,
                             state.getCapacity("A"),
                             state.getCapacity("B")))
        print()
        for i in range(len(messages)):
            print("  {0:2}. {1}".format(i + 1, messages[i]))
        print()
