# Sudoku Solver
![Welcome - Google Chrome 09-08-2021 00_41_14 (2)](https://user-images.githubusercontent.com/71375214/128643467-92095a33-f5d8-43a4-b69a-6fd35cc500c0.png)

This project is done using opencv and CNN model, later was implemented on the Flask web-application . The opencv was used to the preprocess the image, wrap and obtain the sudoku square. This huge square was split into 81 smaller squares(9*9) so that the digit inside each boxes cound be recognised using a digit recognition CNN model. 

# Prediction of digit recognition![Digit Recognition - Google Chrome 09-08-2021 00_49_19](https://user-images.githubusercontent.com/71375214/128643404-32aee26f-1c57-4b03-9554-ea010e6604ce.png)

After finding all the digits these were converted into a array and sent into a sudoku solver which return all the missing values. These values were overlayed on the original image.

![Welcome - Google Chrome 09-08-2021 00_42_24](https://user-images.githubusercontent.com/71375214/128643536-2ab7a5aa-9952-492f-ba6a-d6b1c52f0cfc.png)
![Welcome - Google Chrome 09-08-2021 00_42_39](https://user-images.githubusercontent.com/71375214/128643542-239da8a9-c039-489d-9a96-d96056d7c3a2.png)


