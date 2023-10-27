# Delaunay Triangulation in Python

## Introduction

This repository contains a simple implementation of the [Delaunay triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation) algorithm for two-dimensional points using Python and Numpy.

## Installation

To use this library, you will need to have Python installed on your machine along with pip (the package installer for Python). To install dependencies from source code:

pip install -r requirements.txt


## Usage

The main function is named "delaunay". Here's how it can be used:

from delaunay import delaunay
points = [[0., 0.], [0.5, sqrt(3)/4], [1., 0.]]
simplices = delaunay(points)
print("Simplex:\n", simplices[0])


In this example, we create three points `[(0,0), (0.5,sqrt(3)/4), (1,0)]`, which form an equilateral triangle. The output should print out one or more lines representing each generated simplex.

## Contributing

If you would like to contribute to this project, please feel free to submit pull requests or issues. Any contributions are greatly appreciated!

## License

MIT License Copyright (c) 2023 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
