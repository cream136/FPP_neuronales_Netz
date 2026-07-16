namespace NeuronalesNetz;

/// <summary>
/// Ersatz für die hier benötigten numpy-Operationen: eine dichte Matrix in
/// row-major Speicherung mit den Rechenoperationen, die das Netz braucht.
/// </summary>
public sealed class Matrix
{
    public int Rows { get; }
    public int Cols { get; }
    private readonly double[] _data;

    public Matrix(int rows, int cols)
    {
        Rows = rows;
        Cols = cols;
        _data = new double[rows * cols];
    }

    public double this[int row, int col]
    {
        get => _data[row * Cols + col];
        set => _data[row * Cols + col] = value;
    }

    /// <summary>Spaltenvektor aus einem Array — entspricht numpy.array(x, ndmin=2).T</summary>
    public static Matrix ColumnVector(double[] values)
    {
        var m = new Matrix(values.Length, 1);
        Array.Copy(values, m._data, values.Length);
        return m;
    }

    /// <summary>Normalverteilte Zufallswerte, entspricht numpy.random.normal(mean, stdDev, shape).</summary>
    public static Matrix RandomNormal(Random random, double mean, double stdDev, int rows, int cols)
    {
        var m = new Matrix(rows, cols);
        for (int i = 0; i < m._data.Length; i++)
        {
            m._data[i] = mean + stdDev * NextGaussian(random);
        }
        return m;
    }

    // Box-Muller-Transformation: .NET hat keinen eingebauten Gauß-Generator.
    private static double NextGaussian(Random random)
    {
        double u1 = 1.0 - random.NextDouble(); // (0,1], damit Log(u1) definiert ist
        double u2 = random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }

    /// <summary>Matrixmultiplikation, entspricht numpy.dot(a, b).</summary>
    public static Matrix Dot(Matrix a, Matrix b)
    {
        if (a.Cols != b.Rows)
        {
            throw new ArgumentException($"Dimensionen passen nicht: ({a.Rows}x{a.Cols}) · ({b.Rows}x{b.Cols})");
        }

        var result = new Matrix(a.Rows, b.Cols);
        for (int i = 0; i < a.Rows; i++)
        {
            for (int k = 0; k < a.Cols; k++)
            {
                double aik = a[i, k];
                if (aik == 0.0) continue;
                for (int j = 0; j < b.Cols; j++)
                {
                    result[i, j] += aik * b[k, j];
                }
            }
        }
        return result;
    }

    /// <summary>Multiplikation mit der Transponierten von b, ohne b vorher zu transponieren.</summary>
    public static Matrix DotTransposed(Matrix a, Matrix b)
    {
        if (a.Cols != b.Cols)
        {
            throw new ArgumentException($"Dimensionen passen nicht: ({a.Rows}x{a.Cols}) · ({b.Cols}x{b.Rows})ᵀ");
        }

        var result = new Matrix(a.Rows, b.Rows);
        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < b.Rows; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < a.Cols; k++)
                {
                    sum += a[i, k] * b[j, k];
                }
                result[i, j] = sum;
            }
        }
        return result;
    }

    /// <summary>Multiplikation der Transponierten von a mit b — entspricht numpy.dot(a.T, b).</summary>
    public static Matrix TransposedDot(Matrix a, Matrix b)
    {
        if (a.Rows != b.Rows)
        {
            throw new ArgumentException($"Dimensionen passen nicht: ({a.Cols}x{a.Rows})ᵀ · ({b.Rows}x{b.Cols})");
        }

        var result = new Matrix(a.Cols, b.Cols);
        for (int k = 0; k < a.Rows; k++)
        {
            for (int i = 0; i < a.Cols; i++)
            {
                double aki = a[k, i];
                if (aki == 0.0) continue;
                for (int j = 0; j < b.Cols; j++)
                {
                    result[i, j] += aki * b[k, j];
                }
            }
        }
        return result;
    }

    /// <summary>Elementweise Subtraktion.</summary>
    public static Matrix Subtract(Matrix a, Matrix b)
    {
        EnsureSameShape(a, b);
        var result = new Matrix(a.Rows, a.Cols);
        for (int i = 0; i < result._data.Length; i++)
        {
            result._data[i] = a._data[i] - b._data[i];
        }
        return result;
    }

    /// <summary>Wendet eine Funktion auf jedes Element an.</summary>
    public Matrix Map(Func<double, double> fn)
    {
        var result = new Matrix(Rows, Cols);
        for (int i = 0; i < _data.Length; i++)
        {
            result._data[i] = fn(_data[i]);
        }
        return result;
    }

    /// <summary>
    /// Berechnet elementweise errors * outputs * (1 - outputs) — der Gradiententerm
    /// des Sigmoids, der in beiden Gewichts-Updates vorkommt.
    /// </summary>
    public static Matrix SigmoidGradient(Matrix errors, Matrix outputs)
    {
        EnsureSameShape(errors, outputs);
        var result = new Matrix(errors.Rows, errors.Cols);
        for (int i = 0; i < result._data.Length; i++)
        {
            double o = outputs._data[i];
            result._data[i] = errors._data[i] * o * (1.0 - o);
        }
        return result;
    }

    /// <summary>Addiert delta * scale auf diese Matrix — das In-Place-Update der Gewichte.</summary>
    public void AddScaled(Matrix delta, double scale)
    {
        EnsureSameShape(this, delta);
        for (int i = 0; i < _data.Length; i++)
        {
            _data[i] += delta._data[i] * scale;
        }
    }

    /// <summary>Index des größten Elements — entspricht numpy.argmax.</summary>
    public int ArgMax()
    {
        int best = 0;
        for (int i = 1; i < _data.Length; i++)
        {
            if (_data[i] > _data[best]) best = i;
        }
        return best;
    }

    private static void EnsureSameShape(Matrix a, Matrix b)
    {
        if (a.Rows != b.Rows || a.Cols != b.Cols)
        {
            throw new ArgumentException($"Formen unterschiedlich: ({a.Rows}x{a.Cols}) vs. ({b.Rows}x{b.Cols})");
        }
    }

    public override string ToString()
    {
        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < Rows; i++)
        {
            for (int j = 0; j < Cols; j++)
            {
                if (j > 0) sb.Append(' ');
                sb.Append(this[i, j].ToString("F8", System.Globalization.CultureInfo.InvariantCulture));
            }
            sb.AppendLine();
        }
        return sb.ToString();
    }
}
