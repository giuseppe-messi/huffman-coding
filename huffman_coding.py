"""
Author: Giuseppe Messina
Student ID: S22009501
Project Name: Huffman Coding

This program implements Huffman coding, a widely used data compression algorithm. 
Huffman coding allows for efficient encoding and decoding of data, particularly when 
certain symbols or characters occur more frequently in the input.


The program is structured as follows:

Node Class: The program begins with a Node class, serving as the fundamental building 
block for constructing the Huffman tree. Each node in the tree contains attributes for 
frequency, value, and references to left and right child nodes.

BinHeap Class: To maintain a priority queue of nodes, the program employs a BinHeap class, 
which utilizes a binary heap data structure. This data structure ensures that nodes with 
lower frequencies are placed at the top of the heap, enabling efficient tree construction.

build_huffman_tree Function: This key function takes a text as input and returns a Huffman 
tree data structure. It accomplishes this by constructing a frequency hashtable, 
tallying the occurrence of each character in the given text. Then, it individually inserts 
unique characters into the binary heap, treating each character as a node with a frequency 
value. This foundational step forms the basis for building the Huffman tree. Nodes with lower 
frequencies take precedence in the tree's hierarchy, allowing for efficient encoding and 
decoding of the input text.

encode Function: The encode function uses the generated Huffman tree to encode input text. 
It traverses the tree to find the binary code for each character in the text. The result 
is a binary representation of the input text, which is more efficient for transmission 
and storage.

decode Function: In tandem with the Huffman tree, the decode function reverses the encoding 
process. It traverses the tree according to the encoded bits, reconstructing the original 
text from the binary code.

Interactive Usage: The main loop of the program allows for interactive usage, enabling users 
to input text for encoding. The program then encodes and decodes the text, displaying both 
the encoded and decoded versions. Users have the option to encode additional text or exit 
the program, providing a hands-on experience with Huffman coding. This interactive loop serves 
as a practical demonstration of tree construction, encoding, and decoding.

Efficient Handling of Uniform Characters:

In cases where the Huffman tree consists of a single node, indicating that all characters in 
the encoded text are identical, the decode function offers an efficient mechanism. Instead of 
traversing the tree, it swiftly returns the repeated character followed by its frequency of 
occurrence. This optimized approach ensures exceptional efficiency, even when handling uniform 
characters in the input text.

Time Complexity:
- build_huffman_tree: O(n * log(n))
- encode: O(n * log(n))
- decode: O(n * log(n))

Space Complexity:
- build_huffman_tree: O(n)
- encode: O(n)
- decode: O(1)

"""


class Node:
    """
    Define a class for nodes in the Huffman tree.

    Attributes:
        - freq: The frequency of the symbol associated with this node.
        - value: The symbol value itself.
        - left: A reference to the left child node in the Huffman tree.
        - right: A reference to the right child node in the Huffman tree.
    """

    def __init__(self, frequency, value=None, left=None, right=None):
        """
        Initialize a new node.

        Time Complexity:
        O(1) - Constant time operation because it only initializes a new node.
        Space Complexity: O(1)
        """
        self.freq = frequency
        self.value = value
        self.left = left
        self.right = right

    def update_freq(self):
        """
        Update the frequency of the node by incrementing it by 1.

        Time Complexity:
        O(1) - Constant time operation because it involves a simple increment.
        Space Complexity: O(1)
        """
        self.freq += 1


class BinHeap:
    """
    Define a class for the binary heap data structure, which is used for
    maintaining the priority queue of nodes in Huffman coding.

    Attributes:
        - items: A list that represents the binary heap.
    """

    def __init__(self):
        """
        Initialize a new binary heap with a list holding a '0' in first position.

        Time Complexity:
        O(1) - Constant time operation because it only initializes the list with one value.
        Space Complexity: O(1)
        """
        self.items = [0]

    def __len__(self):
        """
        Return the number of elements in the binary heap.

        Time Complexity: O(1) - Constant time operation because it returns the length of the list.
        Space Complexity: O(1)
        """
        return len(self.items) - 1

    def percolate_up(self):
        """
        Move an element up the heap to maintain the min-heap property.

        Time Complexity: O(log n) - Where 'n' is the size of the heap.
        In the worst case, it has to traverse the height of the binary heap.
        Space Complexity: O(1)
        """
        i = len(self)
        current = self.items[i]
        parent = self.items[i // 2]

        while i // 2 > 0:
            if current.freq < parent.freq:
                parent, current = current, parent
            i = i // 2

    def insert(self, k):
        """
        Insert an element into the binary heap and maintain the min-heap property by percolating it up.

        Args:
            - k: The element to be inserted.

        Time Complexity: O(log n) that comes from the percolate_up() method. Appending the item
        happens in O(1) time.

        Space Complexity: O(1)
        """
        self.items.append(k)
        self.percolate_up()

    def percolate_down(self, i):
        """
        Move an element down the heap to maintain the min-heap property.

        Args:
            - i: The index from which percolation starts.

        Time Complexity: O(log n) - Where 'n' is the size of the heap. In the worst case,
        it has to traverse the height of the binary heap.
        Space Complexity: O(1)
        """
        while i * 2 <= len(self):
            mc = self.min_child(i)
            if self.items[i].freq > self.items[mc].freq:
                self.items[i], self.items[mc] = self.items[mc], self.items[i]
            i = mc

    def min_child(self, i):
        """
        Find the index of the minimum child of a given element in the binary heap.

        Args:
            - i: The index of the element for which the minimum child is found.

        Time Complexity: O(1) - This method only compares two child nodes and determines the minimum.
        Space Complexity: O(1)
        """
        if i * 2 + 1 > len(self):
            return i * 2

        if self.items[i * 2].freq < self.items[i * 2 + 1].freq:
            return i * 2

        return i * 2 + 1

    def delete_min(self):
        """
        Delete and return the minimum element in the binary heap while maintaining the min-heap property.

        Time Complexity: O(log n) - which comes from calling the percolate_down() method.
        Space Complexity: O(1)
        """
        return_value = self.items[1]
        self.items[1] = self.items[len(self)]
        self.items.pop()
        self.percolate_down(1)
        return return_value

    def build_heap(self, alist):
        """
        Build a binary heap from a list of elements by percolating elements down as needed.

        Args:
            - alist: A list of elements to build the heap from.

        Time Complexity: O(n * log n) - Where 'n' is the number of elements in the list.
        Building the heap requires examining each element once, and percolating down
        each element takes O(log n) time.
        Space Complexity: O(1)
        """
        i = len(alist) // 2
        self.items = [0] + alist
        while i > 0:
            self.percolate_down(i)
            i = i - 1


def build_huffman_tree(text):
    """
    Builds and returns a Huffman tree given a text as argument.

    Args:
        - text: A string to be encoded using Huffman coding.

    Time Complexity: O(n * log(n))
    - In the first loop, we iterate through the 'text' to build a frequency dictionary. This loop
      takes O(n) time, where 'n' is the length of the text.
    - In the second loop, we insert all the nodes into a binary heap, which takes O(n * log(n)) time.
    - The third loop constructs the Huffman tree. Since we perform 'n-1' iterations, each involving
      insertions and deletions in a binary heap, the time complexity is O(n * log(n)).
    - The final return statement is a constant-time operation, so it doesn't significantly affect
      the overall time complexity.

    The dominant factor in the time complexity is the O(n * log(n)) operations related to
    building the Huffman tree.

    Space Complexity: O(n)
    - We use a 'frequency' dictionary to store the frequency of each symbol in the 'text.' In the worst
      case, this dictionary can have 'n' unique symbols, so the space complexity is O(n).
    - We also use a binary heap ('bh') to store the nodes. The space complexity of the binary heap is O(n).
    - The space complexity for other variables and function calls is constant or negligible compared to
      the above data structures.

    Therefore, the overall space complexity is O(n).
    """
    frequency = {}

    for symbol in text:
        if symbol not in frequency:
            frequency[symbol] = Node(1, symbol)
        else:
            frequency[symbol].update_freq()

    bh = BinHeap()

    for node in frequency.values():
        bh.insert(node)

    while len(bh) > 1:
        first = bh.delete_min()
        second = bh.delete_min()
        total_freq = first.freq + second.freq
        new = Node(total_freq)
        new.left, new.right = second, first
        bh.insert(new)

    return bh.items[1]


def encode(tree, text):
    """
    Encodes the input text using the provided Huffman tree.

    Args:
        - tree: The Huffman tree used for encoding.
        - text: The text to be encoded.

    Time Complexity: O(n * log(n))
    - 'n' is the length of the input text.
    - For each character in the text, we perform a lookup in the Huffman tree to find its binary code.
      The lookup operation takes O(log(n)) time, where 'n' is the number of nodes in the Huffman tree.
    - Therefore, encoding the entire text takes O(n * log(n)) time.

    Space Complexity: O(n)
    - We create a dictionary 'dic' to store the binary codes for each symbol in the Huffman tree.
      In the worst case, this dictionary can have 'n' unique symbols, so the space complexity is O(n).
    - The 'stack' used for tree traversal can have a maximum of 'n' nodes in it, resulting in an
      additional O(n) space.
    - The 'encoded' string also requires O(n) space to store the encoded text, where 'n' is the length
      of the input text.
    - The space complexity for other variables and function calls is constant or negligible compared to
      the above data structures.

    Special Case Handling:
    - If there is only one unique character in the text (e.g., 'aaa'), it is encoded as the character
      followed by its frequency (e.g., 'a3'). In this case, space complexity may be lower, but in the
      worst case, it remains O(n) due to the 'encoded' string.

    Returns:
        A string representing the encoded version of the input text.
    """
    dic = {}
    stack = [(tree, "")]
    encoded = ""
    while stack:
        node, code = stack.pop()

        if node.left is None and node.right is None:
            dic[node.value] = code
        if node.left:
            stack.append((node.left, code + "0"))
        if node.right:
            stack.append((node.right, code + "1"))

    if len(dic) == 1:
        value = tree.value
        freq = tree.freq
        return f"{value}{freq if freq > 1 else ''}"

    for l in text:
        encoded += dic[l]

    return encoded


def decode(tree, encoded_text):
    """
    Decodes the input encoded text using the provided Huffman tree.

    Args:
        - tree: The Huffman tree used for decoding.
        - encoded_text: The text to be decoded.

    Time Complexity: O(n * log(n))
    - 'n' is the length of the input encoded text.
    - For each bit in the encoded text, we traverse the Huffman tree to find the corresponding symbol.
      The traversal operation takes O(log(n)) time, where 'n' is the number of nodes in the Huffman tree.
    - Therefore, decoding the entire encoded text takes O(n * log(n)) time.

    Space Complexity: O(1)
    - The space complexity of this function is constant (O(1)) as it doesn't use any data structures that
      grow with the input size.
    - The 'decoded' string is the only variable that stores the result, and its size is determined by the
      decoded text, not the input size.

    Special Case Handling:
    - If the Huffman tree consists of a single node (no internal nodes), this indicates that all characters
      in the encoded text are the same. In this case, the function efficiently returns the repeated character
      followed by its frequency (e.g., 'a3') without traversing the tree.

    Returns:
        A string representing the decoded version of the input encoded text.
    """

    if tree.left is None and tree.right is None:
        value = tree.value
        freq = tree.freq
        return f"{value * int(freq)}" if freq > 1 else value

    decoded = ""
    current = tree

    for bit in encoded_text:
        if bit == "0":
            current = current.left
        else:
            current = current.right

        if current.left is None and current.right is None:
            decoded += current.value
            current = tree

    return decoded


while True:
    try:
        text = input("\nWhat text would you like to encode?\n")

        if len(text) == 0:
            raise IndexError

        print(f"\nYou typed: {text}")

        huffman_tree = build_huffman_tree(text)

        encoded_text = encode(huffman_tree, text)
        print(f"\nThat is encoded to: {encoded_text}")

        decoded_text = decode(huffman_tree, encoded_text)
        print(f"\nAnd now it has been decoded back to: {decoded_text}\n")

        res = (
            input("Would you like to encode another text? (yes, no): ").strip().lower()
        )
        if res != "yes":
            print("\nThanks for using this simple program! Bye!\n")
            break

    except IndexError:
        print("\nError: Please type something to be encoded!\n")
