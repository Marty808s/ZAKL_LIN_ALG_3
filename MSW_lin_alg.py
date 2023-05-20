
import matplotlib.pyplot as plt
import numpy as np
import timeit

list_matrix = []
list_liang = []
list_jacobi = []


matice_range = 200
n_iteraci = 10
n = 2

def matice(matice_range,n):
    for w in range(n,matice_range+1):
        n_sloupcu = n
        n_radku = n
        log = True

        while log == True:
            A = np.random.randint(10, size=(n_radku,n_sloupcu))
            b = np.full((n_radku,1),5)


            if np.linalg.det(A) != 0:
                log = False
                print("------------------------------Matice:------------------------------\n", n, "x", n)
                print(A)
                print("Vektor b:\n", b)
                x0 = np.ones(len(A))
                x_jacobi = jacobi(A,b,n_iteraci,x0,dtype=int)


                linalg_starting_time = timeit.default_timer() #zacatek casomiry

                x_linalg = np.linalg.solve(A, b) #build in metoda

                liang_time = 1000*(timeit.default_timer() -  linalg_starting_time) #vypocet casu
                list_liang.append(liang_time) #append casu do listu

                print("LIANG: Time difference :", liang_time, "ms")
                print("Výsledky:")
                print("Jacobi:", x_jacobi)
                print("linalg.solve:", x_linalg)
                print("---------------------------------------------------------------")

                #appending do seznamu
                list_matrix.append(n_radku) #append informaci ohledne velikosti matic

                print("list_liang", list_liang)
                print("list_matrix", list_matrix)

                n+=1
                matice(matice_range,n)
                return

            else:
                log = False
                matice(matice_range,n)
                return

def jacobi(A,b,n_iteraci,x0,dtype = int):
    jacobi_starting_time = timeit.default_timer() #casomira jacobi start

    x = x0
    D = np.diag(A)
    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)

    for i in range(n_iteraci):
        x = (b - np.matmul((L + U),x))/D
        print("iterace:",i, "x=",x)

    jacobi_time = 1000*(timeit.default_timer() - jacobi_starting_time)
    list_jacobi.append(jacobi_time) #casomira jacobi end

    print("JACOBI: Time difference :", jacobi_time, "ms")
    print("list_jacobi", list_jacobi)
    return x


matice(matice_range,n)

#OUTPUT
plt.plot(list_matrix, list_jacobi, 'o-r')
plt.legend(['Jacobi Method'])
plt.plot(list_matrix, list_liang, 'o-g')
plt.title('Built-in Method vs Jacobi Iteration Method (25 iterations)')
plt.ylabel('Time (ms)')
plt.xlabel('Matrix size')
plt.legend(['Jacobi Method','Liang.solve Method'])
plt.grid()


#zjisteni pruseciku
intersection_index = np.argmin(np.abs(np.array(list_jacobi) - np.array(list_liang)))
intersection_x = list_matrix[intersection_index]
intersection_y = list_jacobi[intersection_index]


#zobrazeni krizku v pruseciku
plt.plot(intersection_x, intersection_y, 'k+', markersize=10)


#nastaveni x axis
plt.xlim(min(list_matrix) - 10, max(list_matrix) + 10)
plt.xticks(np.arange(min(list_matrix), max(list_matrix) + 20, 20))

#nasatveni y axis
plt.ylim(0, max(max(list_jacobi), max(list_liang)) + 10)
plt.yticks(np.arange(0, max(max(list_jacobi), max(list_liang)) + 25, 25))

plt.show()
