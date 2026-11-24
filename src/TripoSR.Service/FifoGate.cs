namespace TripoSR.Service;

/// <summary>Runs one piece of work at a time; the others wait in arrival order. <see cref="Busy"/> and <see cref="Queued"/> are what <c>/health</c> reports.</summary>
internal sealed class FifoGate
{
    private readonly object _lock = new();
    private readonly Queue<TaskCompletionSource> _waiters = new();
    private bool _running;

    public bool Busy { get { lock (_lock) return _running; } }

    public int Queued { get { lock (_lock) return _waiters.Count; } }

    public async Task<T> RunAsync<T>(Func<T> work)
    {
        TaskCompletionSource? turn = null;
        lock (_lock)
        {
            if (_running) { turn = new(TaskCreationOptions.RunContinuationsAsynchronously); _waiters.Enqueue(turn); }
            else _running = true;
        }
        if (turn is not null) await turn.Task; // the runner before us hands the turn over without ever clearing "running"
        try { return await Task.Run(work); }
        finally
        {
            lock (_lock)
            {
                if (_waiters.Count > 0) _waiters.Dequeue().SetResult();
                else _running = false;
            }
        }
    }
}
