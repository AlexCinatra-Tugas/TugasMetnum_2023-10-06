from matplotlib import pyplot as plt
import numpy as np

# Gauss-Seidle Method
class GaussSeidelMethod:
    def __init__(self, Matrix, Res):
        self.Matrix = Matrix
        self.Res = Res
        self.root = []
        self.relative_err = []
        self.error = []
    
    def _isAllowed(self):
        return self.Matrix.shape[0] == self.Matrix.shape[1] and self.Matrix.shape[0] == self.Res.shape[0]
    
    def rel_err(self, x_new, x_old):
        return np.abs((x_new - x_old) / x_new) * 100
    
    def _isConverge(self, x_new, x_old, tolerance=0.0001):
        return sum((abs(x_new - x_old) <= tolerance)*1) == x_new.shape[0] and sum((self.rel_err(x_new=x_new, x_old=x_old) < tolerance)*1) == x_new.shape[0]
    
    def err(self, val_new, val_old):
        return np.square(val_new - val_old)
    
    def do(self, x, itteration=5, tol=0.0000001, disp=5):
        if self._isAllowed():
            n = self.Res.shape[0]
            self.root.append(list(x))
            for i in range(itteration):
                for j in range(n):
                    A_star = np.delete(self.Matrix[j], j)
                    x_star = np.delete(x, j)
                    x[j] = (self.Res[j] - A_star @ x_star) / self.Matrix[j][j]
                self.root.append(list(x))
                a = np.array(list(self.root[-1]))
                b = np.array(list(self.root[-2]))
                self.relative_err.append(np.round(self.rel_err(x_new=a, x_old=b), 3))
                self.error.append(np.round(self.err(val_new=self.Matrix @ a, val_old=self.Matrix @ b), 3))
                print(f"{i+1}) x:{np.round(x, disp)} error relative : [{np.round(self.rel_err(x_new=a, x_old=b), disp)}]")
                
                if self._isConverge(x_old=b, x_new=a, tolerance=tol):
                    print(f"SUDAH KONVERGEN, dengan toleransi {tol} pada iterasi ke {i+1}")
                    break
    
    def plot_root(self, save_figure=False):
        roots = np.array(self.root)
        n = np.linspace(0, len(self.root), num=len(self.root))
        plt.title("Plot Dari Nilai X")
        for i in range(roots.shape[1]):
            plt.plot(n, roots[:, i], label=f"x{i} = {np.round(roots[-1][i], 4)}")
        plt.legend(title="Keterangan")
        plt.show()
        if save_figure:
            plt.savefig("Roots-GS.png")
        
    def plot_error(self, save_figure=False):
        rel_errors = np.array(self.relative_err)
        errors = np.array(self.error)
        n = np.linspace(0, len(self.error), num=len(self.error))
        plt.title("Plot Dari Error")
        for i in range(errors.shape[1]):
            plt.plot(n, rel_errors[:, i], label=f"relative error x{i}")
            plt.plot(n, errors[:, i], label=f"Error x{i}")
        plt.legend(title="Keterangan", loc="upper left")
        plt.show()
        if save_figure:
            plt.savefig("Error-GS.png")

# numerical method newton raphson
class NewontRaphson:
    def __init__(self, Matrix_Jacobi, function):
        self.Matrix_Jacobi = Matrix_Jacobi
        self.function = function
        self.root = []
        self.relative_error = []
        self.error = []
    
    def rel_err(self, x_new, x_old):
        state = (x_new == np.zeros(shape=x_new.shape))*1
        n = sum(state)
        if n == x_new.shape[0]:
            ret_val = np.abs((x_new - x_old))
        else:
            ret_val = np.abs((x_new - x_old) / x_new) * 100
        return ret_val

    def err(self, val):
        return np.square(val)
    
    def _isConverge(self, x_new, x_old, tolerance=0.0001):
        s1 = sum((abs(x_new - x_old) <= tolerance)*1) == x_new.shape[0]
        s2 = sum((self.rel_err(x_new=x_new, x_old=x_old) < tolerance)*1) == x_new.shape[0]
        return s1 and s2
    
    def do(self, x, itteration=5, disp=4, tol=0.00001):
        self.root.append(list(x))
        for i in range(itteration):
            if np.linalg.det(self.Matrix_Jacobi(X=x)) != 0:
                x = x - np.linalg.inv(self.Matrix_Jacobi(X=x)) @ self.function(X=x)
                self.root.append(list(x))
                a = np.array(list(self.root[-1]))
                b = np.array(list(self.root[-2]))
                self.relative_error.append(np.round(self.rel_err(x_new=a, x_old=b), 3))
                self.error.append(self.err(val=self.function(X=x)))
                print(f"{i+1}) x:{np.round(x, disp)} error relative : [{np.round(self.rel_err(x_new=a, x_old=b), disp)}]")
                if self._isConverge(x_new=a, x_old=b):
                    print(f"SUDAH KONVERGEN, dengan toleransi {tol} pada iterasi ke {i+1}")
                    break
            else:
                print("The Determinant of jacobian Matrix is 0, can't continue")
                print(self.Matrix_Jacobi(X=x))
                break
    
    def plot_root(self, save_figure=False):
        roots = np.array(self.root)
        n = np.linspace(0, len(self.root), num=len(self.root))
        plt.title("Plot Dari Nilai X")
        for i in range(roots.shape[1]):
            plt.plot(n, roots[:, i], label=f"X{i} = {np.round(roots[-1][i], 4)}")
        plt.legend(title="Keterangan")
        plt.show()
        if save_figure:
            plt.savefig("Roots-NR.png")
    
    def plot_error(self, save_figure=False):
        rel_errors = np.array(self.relative_error)
        errors = np.array(self.error)
        n = np.linspace(0, len(self.error), num=len(self.error))
        plt.title("Plot Dari Error")
        for i in range(errors.shape[1]):
            plt.plot(n, rel_errors[:, i], label=f"relative error x{i}")
            plt.plot(n, errors[:, i], label=f"Error f{i}")
        plt.legend(title="Keterangan", loc="upper left")
        plt.show()
        if save_figure:
            plt.savefig("Error-NR.png")

# numerical method Secant
class SectonMethod:
    def __init__(self, Matrix_Jacobi, function):
        self.Matrix_Jacobi = Matrix_Jacobi
        self.function = function
        self.root = []
        self.relative_error = []
        self.error = []
    
    def rel_err(self, x_new, x_old):
        state = (x_new == np.zeros(shape=x_new.shape))*1
        n = sum(state)
        if n == x_new.shape[0]:
            ret_val = np.abs((x_new - x_old))
        else:
            ret_val = np.abs((x_new - x_old) / x_new) * 100
        return ret_val

    def err(self, val):
        return np.square(val)
    
    def _isConverge(self, x_new, x_old, tolerance=0.00001):
        s1 = sum((abs(x_new - x_old) <= tolerance)*1) == x_new.shape[0]
        s2 = sum((self.rel_err(x_new=x_new, x_old=x_old) < tolerance)*1) == x_new.shape[0]
        return s1 and s2

    
    def do(self, x, itteration=5, disp=4, tol=0.0001):
        self.root.append(list(x))
        for i in range(itteration):
            if np.linalg.det(self.Matrix_Jacobi(X=x)) != 0:
                x = x - np.linalg.inv(self.Matrix_Jacobi(X=x)) @ self.function(X=x)
                self.root.append(list(x))
                a = np.array(list(self.root[-1]))
                b = np.array(list(self.root[-2]))
                self.relative_error.append(np.round(self.rel_err(x_new=a, x_old=b), 3))
                self.error.append(self.err(val=self.function(X=x)))
                print(f"{i+1}) x:{np.round(x, disp)} error relative : [{np.round(self.rel_err(x_new=a, x_old=b), disp)}]")
                if self._isConverge(x_new=a, x_old=b):
                    print(f"SUDAH KONVERGEN, dengan toleransi {tol} pada iterasi ke {i+1}")
                    break
            else:
                print("The Determinant of jacobian Matrix is 0, can't continue")
                print(self.Matrix_Jacobi(X=x))
                break
    
    def plot_root(self, save_figure=False):
        roots = np.array(self.root)
        n = np.linspace(0, len(self.root), num=len(self.root))
        plt.title("Plot Dari Nilai X")
        for i in range(roots.shape[1]):
            plt.plot(n, roots[:, i], label=f"X{i} = {np.round(roots[-1][i], 4)}")
        plt.legend(title="Keterangan")
        plt.show()
        if save_figure:
            plt.savefig("Roots-NR.png")
    
    def plot_error(self, save_figure=False):
        rel_errors = np.array(self.relative_error)
        errors = np.array(self.error)
        n = np.linspace(0, len(self.error), num=len(self.error))
        plt.title("Plot Dari Error")
        for i in range(errors.shape[1]):
            plt.plot(n, rel_errors[:, i], label=f"relative error x{i}")
            plt.plot(n, errors[:, i], label=f"Error f{i}")
        plt.legend(title="Keterangan", loc="upper left")
        plt.show()
        if save_figure:
            plt.savefig("Error-NR.png")
