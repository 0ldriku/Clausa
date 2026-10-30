// Clausa.exe: the Windows app's launcher. packaging/build_windows.py compiles this file
// with the .NET Framework's csc.exe (part of every Windows 10 and 11) and fills in the
// numbers in {{...}}. C# 5, which that csc.exe understands.
//
// The folder holds Clausa.exe, envs\ (until First start: one tar per environment), cache\,
// app\ (the code) and logs\. On every start this:
// 1. checks the folder: its path must leave room for the longest path inside it (Windows'
//    limit of 260 characters, with long paths off as they are by default), it must not be a
//    network folder or in a folder that syncs to the cloud, Clausa must be able to write there,
//    and the WebView2 runtime (the app window's web engine) must be installed. Otherwise a
//    message says what to do;
// 2. until First start has finished (envs\.first-start-done), runs it: a small "Preparing Clausa"
//    window while this unpacks the first environment (qwen3-asr, piped into Windows' tar.exe; the
//    tar is deleted once it is unpacked whole), then the First start window
//    (runner/first_start_window.py, with that environment's Python), which runs
//    app\packaging\windows\first_start.py (the other environments, PyTorch and the models, loading
//    the libraries once, the marker) and shows its progress. Both write First start's events into
//    logs\first-start-events.jsonl (docs/development.md, "The app zips"). When the window ends with "Open Clausa", this opens the app; if the window cannot
//    open, First start runs in a console window as before;
// 3. otherwise starts the app: envs\qwen3-asr\python.exe -m runner.app in app\, in a console
//    without a window (the stages it starts inherit that console, so none opens a window), its
//    output in logs\console.log. If the app ends with an error, a message points to that file.
//    A second start while the app runs brings its window to the front.
// Its own messages go to logs\launcher.log.
//
// For tests (no person at the screen): CLAUSA_LAUNCHER_TEST=1 writes the messages to standard
// output instead of showing them, uses the console it was given (none is made, nothing waits for
// Enter), shows no window (First start's Python part runs with its output on standard output) and
// keeps the events file; CLAUSA_FIRST_START_NO_OPEN=1 ends after First start instead of opening
// the app.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Windows.Forms;
using Microsoft.Win32;

[assembly: System.Reflection.AssemblyTitle("Clausa")]
[assembly: System.Reflection.AssemblyProduct("Clausa")]
[assembly: System.Reflection.AssemblyCopyright("© 2026 Jialiang Lu and Reiko Sato")]
[assembly: System.Reflection.AssemblyVersion("{{VERSION4}}")]
[assembly: System.Reflection.AssemblyFileVersion("{{VERSION4}}")]
[assembly: System.Reflection.AssemblyInformationalVersion("{{VERSION}}")]

static class Launcher
{
    // The longest path inside the folder, relative to it (measured by the build), and so the
    // longest the folder's own path may be: 259 characters in all.
    const int LongestInside = {{LONGEST_INSIDE}};
    const int MaxRoot = 259 - LongestInside;
    static readonly string[] Envs = { "qwen3-asr", "mfa", "ja-electra310" };
    const string Title = "Clausa";

    static string Root, LogPath;
    static readonly bool Testing = Environment.GetEnvironmentVariable("CLAUSA_LAUNCHER_TEST") == "1";

    [DllImport("kernel32.dll")] static extern bool AllocConsole();
    [DllImport("kernel32.dll")] static extern bool FreeConsole();
    [DllImport("kernel32.dll")] static extern bool SetConsoleTitle(string title);
    [DllImport("user32.dll")] static extern bool SetForegroundWindow(IntPtr hwnd);
    [DllImport("user32.dll")] static extern bool ShowWindow(IntPtr hwnd, int cmd);
    [DllImport("user32.dll")] static extern bool IsIconic(IntPtr hwnd);
    [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr hwnd);
    [DllImport("user32.dll")] static extern int GetWindowText(IntPtr hwnd, StringBuilder text, int count);
    [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr hwnd, out uint pid);
    delegate bool EnumWindowsProc(IntPtr hwnd, IntPtr lParam);
    [DllImport("user32.dll")] static extern bool EnumWindows(EnumWindowsProc proc, IntPtr lParam);
    [DllImport("user32.dll")] static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] static extern bool IsWindow(IntPtr hwnd);
    [DllImport("kernel32.dll")] static extern IntPtr GetConsoleWindow();
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern int GetClassName(IntPtr hwnd, StringBuilder name, int count);

    static string ClassOf(IntPtr hwnd)
    {
        StringBuilder name = new StringBuilder(64);
        GetClassName(hwnd, name, name.Capacity);
        return name.ToString();
    }

    [STAThread]
    static int Main()
    {
        SetProcessDPIAware();  // sharp text in the message boxes and the Preparing window
        Root = Path.GetDirectoryName(Application.ExecutablePath);
        LogPath = Path.Combine(Root, "logs", "launcher.log");
        try
        {
            return Run();
        }
        catch (Exception e)
        {
            Log("unexpected error: " + e);
            Message("Clausa could not start (" + e.Message + "). Details are in logs\\launcher.log in the Clausa " +
                    "folder; please send that file with a description of the problem.", true);
            return 1;
        }
    }

    static int Run()
    {
        string problem = CheckFolder();
        if (problem != null)
        {
            Log("refused: " + problem);
            Message(problem, true);
            return 1;
        }
        bool created;
        // One Clausa per folder: a second start brings the first one's window to the front.
        string hash = BitConverter.ToString(SHA1.Create().ComputeHash(Encoding.UTF8.GetBytes(Root.ToLowerInvariant())))
                                  .Replace("-", "").Substring(0, 16);
        using (Mutex mutex = new Mutex(true, "Local\\clausa-" + hash, out created))
        {
            if (!created)
            {
                if (!BringToFront())
                    Message("Clausa is already running (or its First start is: see its window).", false);
                return 0;
            }
            if (!Prepared())
            {
                int done = FirstStart();
                if (done != Open)
                    return done == Failed ? 1 : 0;
                if (Environment.GetEnvironmentVariable("CLAUSA_FIRST_START_NO_OPEN") == "1")
                {
                    Log("ready; CLAUSA_FIRST_START_NO_OPEN is set: not opening Clausa");
                    if (Testing) Console.WriteLine("Ready (CLAUSA_FIRST_START_NO_OPEN is set: not opening Clausa).");
                    return 0;
                }
            }
            return StartApp();
        }
    }

    // ------------------------------------------------------------------ checks

    static string CheckFolder()
    {
        if (Root.StartsWith("\\\\"))
            return "Clausa cannot run from a network folder (" + Root + "). Copy the Clausa folder to this " +
                   "PC, to a short path such as C:\\Clausa, and start it from there.";
        string odd = NotInAnsiCodePage(Root);
        if (odd != null)
            return "The Clausa folder's path has characters that parts of Clausa cannot read with this PC's " +
                   "language setting (" + odd + "). Move the Clausa folder to a path without them, such as " +
                   "C:\\Clausa, and start it from there.\n\nIt is now in " + Root;
        if (Root.Length > MaxRoot)
            return "The Clausa folder's path is too long for Windows: it has " + Root.Length + " characters, and " +
                   "the files inside need it to have at most " + MaxRoot + ". Move the Clausa folder to a shorter " +
                   "path, such as C:\\Clausa, and start it from there.\n\nIt is now in " + Root;
        string lower = Root.ToLowerInvariant() + "\\";
        foreach (string v in new string[] { "OneDrive", "OneDriveConsumer", "OneDriveCommercial" })
        {
            string d = Environment.GetEnvironmentVariable(v);
            if (!string.IsNullOrEmpty(d) && lower.StartsWith(d.TrimEnd('\\').ToLowerInvariant() + "\\"))
                return Cloud();
        }
        foreach (string part in new string[] { "\\dropbox\\", "\\google drive\\", "\\my drive\\", "\\icloud drive\\",
                                               "\\iclouddrive\\" })
            if (lower.Contains(part))
                return Cloud();
        try
        {
            Directory.CreateDirectory(Path.Combine(Root, "logs"));
            string probe = Path.Combine(Root, "logs", ".write-test");
            File.WriteAllText(probe, "ok");
            File.Delete(probe);
        }
        catch (Exception e)
        {
            return "Clausa cannot write in its folder (" + Root + "): " + e.Message + "\n\nMove the Clausa folder to " +
                   "a folder of your own, such as C:\\Clausa, and start it from there.";
        }
        if (WebView2Version() == null)
            return "Clausa's window needs the Microsoft Edge WebView2 Runtime, which this computer does not have. " +
                   "Windows 11 includes it, and Windows 10 gets it with Microsoft Edge. You can install it from " +
                   "Microsoft: https://go.microsoft.com/fwlink/p/?LinkId=2124703 (the \"Evergreen Bootstrapper\"). " +
                   "Then start Clausa again.";
        return null;
    }

    // Parts of the pipeline open files through Windows' ANSI code page, not Unicode (nagisa's model
    // loader in Stage 1, Kaldi in Stage 2): a path with a character that code page lacks cannot be
    // opened (a folder named テスト on an English Windows: Stage 1 stopped). Returns those characters.
    static string NotInAnsiCodePage(string path)
    {
        Encoding ansi = Encoding.GetEncoding(Encoding.Default.CodePage, new EncoderReplacementFallback("\u0000"),
                                             new DecoderReplacementFallback("?"));
        StringBuilder odd = new StringBuilder();
        foreach (char c in path)
        {
            byte[] one = ansi.GetBytes(c.ToString());
            if (one.Length == 1 && one[0] == 0 && odd.ToString().IndexOf(c) < 0) odd.Append(c);
        }
        return odd.Length > 0 ? odd.ToString() : null;
    }

    static string Cloud()
    {
        return "The Clausa folder is in a folder that syncs to the cloud (OneDrive, Dropbox, Google Drive or " +
               "iCloud): syncing its 15 GB is slow and can remove files Clausa needs. Move the Clausa folder to a " +
               "folder that does not sync, such as C:\\Clausa, and start it from there.\n\nIt is now in " + Root;
    }

    static string WebView2Version()
    {
        // Where the WebView2 Runtime registers itself (Microsoft's documentation): per machine, then per user.
        const string Id = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}";
        string[] keys = { "HKEY_LOCAL_MACHINE\\SOFTWARE\\WOW6432Node\\Microsoft\\EdgeUpdate\\Clients\\" + Id,
                          "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\EdgeUpdate\\Clients\\" + Id,
                          "HKEY_CURRENT_USER\\Software\\Microsoft\\EdgeUpdate\\Clients\\" + Id };
        foreach (string key in keys)
        {
            object v = Registry.GetValue(key, "pv", null);
            if (v != null && v.ToString() != "" && v.ToString() != "0.0.0.0")
                return v.ToString();
        }
        return null;
    }

    static bool Prepared()
    {
        return File.Exists(Path.Combine(Root, "envs", ".first-start-done")) &&
               File.Exists(Path.Combine(Root, "envs", "qwen3-asr", "python.exe")) &&
               File.Exists(Path.Combine(Root, "app", "runner", "app.py"));
    }

    // ------------------------------------------------------------------ First start

    const int Open = 0, End = 1, Failed = 2;  // what FirstStart() leads to
    static string EventsPath { get { return Path.Combine(Root, "logs", "first-start-events.jsonl"); } }
    static string WindowPidPath { get { return Path.Combine(Root, "logs", "first-start-window.pid"); } }
    static StreamWriter WindowOutput;
    // The console First start runs in where its window cannot open: a second start brings it to the front.
    static string ConsoleHandlePath { get { return Path.Combine(Root, "logs", "first-start-console.hwnd"); } }
    const int WindowWait = 120;  // seconds the First start window may take to load its page

    // Returns Open (First start is done: open the app), End (closed before it was done, or after it
    // without "Open Clausa") or Failed.
    static int FirstStart()
    {
        Log("First start in " + Root);
        string run = ((long)(DateTime.UtcNow - new DateTime(1970, 1, 1)).TotalSeconds).ToString(CultureInfo.InvariantCulture) +
                     "-" + Process.GetCurrentProcess().Id.ToString(CultureInfo.InvariantCulture);
        if (Testing)
        {
            string problem = UnpackFirst(run, null, null);
            if (problem != null)
                return FailConsole(problem);
            return RunDriverInConsole(run);
        }

        Application.EnableVisualStyles();
        Preparing form = new Preparing();
        string failure = null;
        Process window = null;
        bool shown = false;
        Thread worker = new Thread(delegate()
        {
            try
            {
                failure = UnpackFirst(run, form, delegate(long done, long total) { form.Unpacking(done, total); });
                if (failure != null || form.Cancelled) return;
                form.Opening();
                window = StartWindow(run);
                // The Preparing window stays until the First start window is there (it writes its
                // process id once its page has loaded), or that one has ended. A window whose page
                // never loads (WebView2 failing to start: pywebview then keeps an empty window open)
                // is ended after WindowWait seconds, and First start goes on in a console.
                DateTime began = DateTime.Now;
                while (true)
                {
                    if (File.Exists(WindowPidPath)) { shown = true; break; }
                    if (window.HasExited || form.Cancelled) break;
                    if ((DateTime.Now - began).TotalSeconds > WindowWait)
                    {
                        Log("the First start window's page did not load within " + WindowWait + " s: ended");
                        KillTree(window);
                        break;
                    }
                    Thread.Sleep(200);
                }
            }
            catch (Exception e)
            {
                Log("First start (Preparing): " + e);
                failure = "First start could not begin (" + e.Message + "). Start Clausa again; if it happens again, " +
                          "please send logs\\launcher.log.";
            }
            finally
            {
                form.Finish();
            }
        });
        worker.IsBackground = true;
        form.Shown += delegate { worker.Start(); };
        Application.Run(form);
        if ((worker.ThreadState & System.Threading.ThreadState.Unstarted) == 0)
            worker.Join();
        if (failure != null)
        {
            Log("Stopped: " + failure);
            AppendFirstStartLog("Stopped: " + failure);
            Message(failure, true);
            return Failed;
        }
        // Once the First start window is there, it decides (a click on the Preparing window's close
        // button in the same moment changes nothing); before, closing the Preparing window stops.
        if (window == null || (!shown && form.Cancelled))
        {
            if (window != null && !window.HasExited)
                KillTree(window);
            Log("First start: the Preparing window was closed");
            return End;
        }
        window.WaitForExit();
        int code = window.ExitCode;
        window.Dispose();
        lock (WindowOutput) WindowOutput.Dispose();
        Log("First start window ended with " + code + (shown ? "" : " (its page never loaded)"));
        if (code == 0 || code == 3)
        {
            if (!Prepared())
            {
                Message("First start did not finish. Start Clausa again; it continues where it stopped.", true);
                return Failed;
            }
            DeleteEvents();
            return code == 0 ? Open : End;
        }
        if (code == 1 && shown)  // closed before done: the next start continues
            return End;
        if (code == 10 || !shown)  // the window could not open (pywebview), or Python failed before it did
        {
            AllocConsole();
            SetConsoleTitle("Clausa: first start");
            Console.OutputEncoding = Encoding.UTF8;
            try { File.WriteAllText(ConsoleHandlePath, GetConsoleWindow().ToInt64().ToString(CultureInfo.InvariantCulture)); }
            catch (Exception) { }
            int rc = RunDriverInConsole(run);
            try { File.Delete(ConsoleHandlePath); } catch (Exception) { }
            // The console closes here: left open beside the app, closing it would end this launcher
            // (and its hold on the one-Clausa-per-folder mutex), not the app.
            if (rc == Open)
                FreeConsole();
            return rc;
        }
        Message("The First start window closed unexpectedly (exit " + code + "). Start Clausa again; it continues " +
                "where it stopped. Details are in logs\\first-start-window.log in the Clausa folder.", true);
        return Failed;
    }

    // The first environment, whose Python runs the rest: envs\qwen3-asr.tar into envs\qwen3-asr,
    // the tar piped into tar.exe so that the bytes read are the progress. The tar is deleted only
    // once it is unpacked whole: while it is there, envs\qwen3-asr is not trusted (an interrupted
    // First start left it half made). Returns null, or what to tell the person.
    static string UnpackFirst(string run, Preparing form, Action<long, long> progress)
    {
        string envs = Path.Combine(Root, "envs"), name = Envs[0];
        string archive = Path.Combine(envs, name + ".tar"), target = Path.Combine(envs, name);
        // This run's start, always: until the driver's own arrives, the window shows this one, not the
        // end of an earlier run (whose error would offer Try again).
        Emit("start", "run", run, "platform", "windows", "version", "{{VERSION}}", "steps", new Raw(
                 "[{\"id\": \"unpack\", \"label\": \"Unpacking the app\"}, {\"id\": \"download\", \"label\": " +
                 "\"Downloading the speech models\"}, {\"id\": \"prepare\", \"label\": \"Preparing (only once)\"}]"),
             "pid", Process.GetCurrentProcess().Id, "resumed", File.Exists(EventsPath));
        Emit("step", "step", "unpack", "status", "running");
        if (!File.Exists(archive))
        {
            if (File.Exists(Path.Combine(target, "python.exe")))
                return null;
            return "Part of the app is missing (neither envs\\" + name + ".tar nor envs\\" + name + " is there). " +
                   "Please extract the download again.";
        }
        string tar = Path.Combine(Environment.SystemDirectory, "tar.exe");
        if (!File.Exists(tar))
            return "Windows' tar.exe was not found (" + tar + "). Clausa needs Windows 10 version 1803 or later.";
        long size = new FileInfo(archive).Length, need = size + (2L << 30), free = long.MaxValue;
        try { free = new DriveInfo(Path.GetPathRoot(envs)).AvailableFreeSpace; }
        catch (Exception e) { Log("free space unknown (" + e.Message + "): unpacking anyway"); }  // e.g. a network path
        if (free < need)
        {
            Log("not enough space for " + name + ": " + free + " bytes free, " + need + " needed");
            return "There is not enough free disk space. Unpacking needs about " + Gb(need, true) + " GB free on this " +
                   "disk; there are " + Gb(free, false) + " GB. Free some space (you can delete the downloaded zip), " +
                   "then start Clausa again.";
        }
        string[] labels = { "The app and speech recognition", "Word alignment", "Japanese grammar" };
        for (int i = 1; i < Envs.Length; i++)
        {
            string other = Path.Combine(envs, Envs[i] + ".tar");
            if (File.Exists(other))
                Emit("item", "step", "unpack", "item", Envs[i], "label", labels[i], "total", new FileInfo(other).Length,
                     "status", "waiting");
        }
        Emit("item", "step", "unpack", "item", name, "label", labels[0], "total", size, "status", "running", "done", 0L);
        if (Testing) Console.WriteLine("Unpacking the app (" + Gb(size, true) + " GB)...");
        DateTime start = DateTime.Now;
        try
        {
            if (Directory.Exists(target))
                Retry(delegate { Directory.Delete(target, true); });
        }
        catch (Exception e)
        {
            Log("cannot remove the unfinished " + target + ": " + e);
            return "Clausa cannot write in envs\\ (" + e.Message + "). Is a file in envs\\" + name + " open? Start " +
                   "Clausa again.";
        }
        ProcessStartInfo psi = new ProcessStartInfo(tar, "-xf - -C " + Quote(envs));
        psi.WorkingDirectory = envs;
        psi.UseShellExecute = false;
        psi.CreateNoWindow = true;
        psi.RedirectStandardInput = psi.RedirectStandardOutput = psi.RedirectStandardError = true;
        Log("run: " + tar + " " + psi.Arguments + " < " + archive);
        StringBuilder said = new StringBuilder();
        long done = 0;
        int code;
        using (Process p = new Process())
        {
            p.StartInfo = psi;
            DataReceivedEventHandler keep = delegate(object sender, DataReceivedEventArgs a)
            {
                if (a.Data != null) lock (said) said.AppendLine(a.Data);
            };
            p.OutputDataReceived += keep;
            p.ErrorDataReceived += keep;
            p.Start();
            p.BeginOutputReadLine();
            p.BeginErrorReadLine();
            Stream into = p.StandardInput.BaseStream;
            DateTime last = DateTime.MinValue;
            using (FileStream src = new FileStream(archive, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20))
            {
                byte[] buffer = new byte[4 << 20];
                int n;
                while ((n = src.Read(buffer, 0, buffer.Length)) > 0)
                {
                    if (form != null && form.Cancelled)
                    {
                        try { p.Kill(); } catch (Exception) { }
                        p.WaitForExit();
                        return null;
                    }
                    try { into.Write(buffer, 0, n); }
                    catch (IOException) { break; }  // tar.exe ended early (a full disk, say): its exit code says so
                    done += n;
                    if ((DateTime.Now - last).TotalSeconds >= 0.25)
                    {
                        last = DateTime.Now;
                        Emit("progress", "step", "unpack", "item", name, "done", done);
                        if (progress != null) progress(done, size);
                    }
                }
            }
            try { into.Close(); } catch (IOException) { }
            p.WaitForExit();
            code = p.ExitCode;
        }
        if (said.Length > 0)
            AppendFirstStartLog("tar.exe: " + said.ToString().TrimEnd());
        if (code != 0 || !File.Exists(Path.Combine(target, "python.exe")))
        {
            string problem = "The app could not be unpacked (tar.exe ended with " + code + "; is the disk full?). " +
                             "Free some space, then start Clausa again.";
            Emit("error", "step", "unpack", "kind", "disk", "message", problem, "again", false);
            return problem;
        }
        Emit("progress", "step", "unpack", "item", name, "done", done);
        if (progress != null) progress(done, size);
        try
        {
            Retry(delegate { File.Delete(archive); });
        }
        catch (Exception e)
        {
            Log("cannot delete " + archive + ": " + e);
            return "envs\\" + name + ".tar could not be deleted (" + e.Message + "). Start Clausa again.";
        }
        // Not `done` yet: First start's Python part runs conda-unpack in it first.
        Log(name + " unpacked in " + (int)(DateTime.Now - start).TotalSeconds + " s");
        return null;
    }

    // The First start window, which runs first_start.py as its driver; its own output goes to
    // logs\first-start-window.log.
    static Process StartWindow(string run)
    {
        string python = Path.Combine(Root, "envs", "qwen3-asr", "python.exe");
        string script = Path.Combine(Root, "app", "packaging", "windows", "first_start.py");
        ProcessStartInfo psi = new ProcessStartInfo(python, "-m runner.first_start_window --root " + Quote(Root) +
            " --driver " + Quote(python) + " -B -s " + Quote(script) + " " + Quote(Root));
        psi.WorkingDirectory = Path.Combine(Root, "app");
        psi.UseShellExecute = false;
        psi.CreateNoWindow = true;
        psi.RedirectStandardOutput = psi.RedirectStandardError = true;
        psi.StandardOutputEncoding = psi.StandardErrorEncoding = Encoding.UTF8;
        Dictionary<string, string> env = AppEnv();
        env["CLAUSA_FIRST_START_RUN"] = run;
        Apply(psi, env);
        try { File.Delete(WindowPidPath); } catch (Exception) { }  // a window that was ended hard left it
        try { File.Delete(ConsoleHandlePath); } catch (Exception) { }  // so did a console fallback
        Log("run: " + python + " " + psi.Arguments);
        StreamWriter output = new StreamWriter(new FileStream(Path.Combine(Root, "logs", "first-start-window.log"),
            FileMode.Create, FileAccess.Write, FileShare.ReadWrite), new UTF8Encoding(false));
        output.AutoFlush = true;
        WindowOutput = output;
        Process p = new Process();
        p.StartInfo = psi;
        DataReceivedEventHandler copy = delegate(object sender, DataReceivedEventArgs a)
        {
            // (a line after the file was closed, or a full disk, must not end Clausa.exe)
            if (a.Data != null) lock (output) try { output.WriteLine(a.Data); } catch (Exception) { }
        };
        p.OutputDataReceived += copy;
        p.ErrorDataReceived += copy;
        p.Start();
        p.BeginOutputReadLine();
        p.BeginErrorReadLine();
        return p;
    }

    // First start's Python part in a console (tests, or where the window cannot open).
    static int RunDriverInConsole(string run)
    {
        Console.WriteLine("Clausa: first start");
        Console.WriteLine("Do not move or rename the Clausa folder, and do not close this window, until it says Done.");
        Console.WriteLine();
        string python = Path.Combine(Root, "envs", "qwen3-asr", "python.exe");
        string script = Path.Combine(Root, "app", "packaging", "windows", "first_start.py");
        Dictionary<string, string> env = AppEnv();
        env["CLAUSA_FIRST_START_RUN"] = run;
        int rc = Exec(python, "-B -s " + Quote(script) + " " + Quote(Root), Path.Combine(Root, "app"), env, true);
        if (rc != 0)
        {
            Log("first_start.py ended with " + rc);
            Console.WriteLine();
            Console.WriteLine("Details are in logs\\first-start.log in the Clausa folder. If you cannot fix this, please");
            Console.WriteLine("send that file with a description of the problem.");
            Pause();
            return Failed;
        }
        if (!Testing)
            DeleteEvents();
        Console.WriteLine();
        Console.WriteLine("Opening Clausa. From now on, double-click Clausa.exe to start it.");
        if (!Testing)
            Thread.Sleep(2000);
        return Open;
    }

    // A file a virus scanner reads just now cannot be deleted: a few tries, half a second apart.
    static void Retry(MethodInvoker action)
    {
        for (int i = 1; ; i++)
        {
            try { action(); return; }
            catch (IOException) { if (i == 5) throw; }
            catch (UnauthorizedAccessException) { if (i == 5) throw; }
            Thread.Sleep(500);
        }
    }

    static int FailConsole(string message)
    {
        Log("Stopped: " + message);
        AppendFirstStartLog("Stopped: " + message);
        Console.WriteLine();
        Console.WriteLine("Stopped: " + message);
        return Failed;
    }

    static void Pause()
    {
        if (Testing) return;
        Console.WriteLine();
        Console.Write("Press Enter to close this window. ");
        try { Console.ReadLine(); } catch (Exception) { }
    }

    // First start is done and its window closed: its events are no longer needed.
    static void DeleteEvents()
    {
        try { File.Delete(EventsPath); } catch (Exception e) { Log("cannot delete the events file: " + e.Message); }
    }

    static void KillTree(Process p)
    {
        try
        {
            ProcessStartInfo psi = new ProcessStartInfo(Path.Combine(Environment.SystemDirectory, "taskkill.exe"),
                                                        "/T /F /PID " + p.Id.ToString(CultureInfo.InvariantCulture));
            psi.UseShellExecute = false;
            psi.CreateNoWindow = true;
            using (Process k = Process.Start(psi)) k.WaitForExit(10000);
        }
        catch (Exception e) { Log("taskkill: " + e.Message); }
    }

    // ------------------------------------------------------------------ First start's events

    // A JSON value written as it is.
    sealed class Raw
    {
        public readonly string Json;
        public Raw(string json) { Json = json; }
    }

    // One event, appended as one line (runner/first_start_events.py): UTF-8 without
    // a byte order mark, numbers the JSON way whatever the system's language, strings escaped;
    // the file shared, so the window can read it meanwhile.
    static void Emit(string name, params object[] fields)
    {
        double t = (DateTime.UtcNow - new DateTime(1970, 1, 1)).TotalSeconds;
        StringBuilder line = new StringBuilder("{\"t\": " + t.ToString("0.000", CultureInfo.InvariantCulture) +
                                               ", \"event\": " + Json(name));
        for (int i = 0; i + 1 < fields.Length; i += 2)
            line.Append(", ").Append(Json((string)fields[i])).Append(": ").Append(Json(fields[i + 1]));
        line.Append("}\n");
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(EventsPath));
            byte[] bytes = new UTF8Encoding(false).GetBytes(line.ToString());
            using (FileStream f = new FileStream(EventsPath, FileMode.Append, FileAccess.Write,
                                                 FileShare.ReadWrite | FileShare.Delete))
                f.Write(bytes, 0, bytes.Length);
        }
        catch (Exception e) { Log("cannot write an event: " + e.Message); }
    }

    static string Json(object v)
    {
        if (v == null) return "null";
        if (v is Raw) return ((Raw)v).Json;
        if (v is bool) return (bool)v ? "true" : "false";
        if (v is int || v is long) return Convert.ToInt64(v).ToString(CultureInfo.InvariantCulture);
        StringBuilder s = new StringBuilder("\"");
        foreach (char c in v.ToString())
        {
            if (c == '"') s.Append("\\\"");
            else if (c == '\\') s.Append("\\\\");
            else if (c < ' ') s.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
            else s.Append(c);
        }
        return s.Append('"').ToString();
    }

    // ------------------------------------------------------------------ the Preparing window

    // A small window in the app's colours while Clausa.exe unpacks the first environment and the
    // First start window starts (about half a minute). Closing it stops First start.
    sealed class Preparing : Form
    {
        [DllImport("dwmapi.dll")] static extern int DwmSetWindowAttribute(IntPtr hwnd, int attr, ref int value, int size);

        readonly Label head = new Label(), detail = new Label(), hint = new Label();
        readonly Bar bar = new Bar();
        volatile bool cancelled, closing;
        public bool Cancelled { get { return cancelled; } }

        public Preparing()
        {
            float k;
            using (Graphics g = CreateGraphics()) k = g.DpiX / 96f;
            Text = Title;
            try { Icon = System.Drawing.Icon.ExtractAssociatedIcon(Application.ExecutablePath); } catch (Exception) { }
            FormBorderStyle = FormBorderStyle.FixedSingle;
            MaximizeBox = false;
            StartPosition = FormStartPosition.CenterScreen;
            AutoScaleMode = AutoScaleMode.None;
            BackColor = Color.FromArgb(0x19, 0x19, 0x19);
            ClientSize = new System.Drawing.Size((int)(440 * k), (int)(150 * k));
            head.Text = "Preparing Clausa\u2026";
            head.Font = new System.Drawing.Font("Segoe UI Semibold", 12f);
            head.ForeColor = Color.FromArgb(0xee, 0xee, 0xee);
            head.SetBounds((int)(24 * k), (int)(22 * k), (int)(392 * k), (int)(26 * k));
            detail.Text = "First start: unpacking the app\u2026";
            detail.Font = new System.Drawing.Font("Segoe UI", 9.75f);
            detail.ForeColor = Color.FromArgb(0xb4, 0xb4, 0xb4);
            detail.SetBounds((int)(24 * k), (int)(54 * k), (int)(392 * k), (int)(22 * k));
            bar.SetBounds((int)(24 * k), (int)(86 * k), (int)(392 * k), Math.Max(4, (int)(6 * k)));
            hint.Text = "Closing this window stops First start; the next start continues it.";
            hint.Font = new System.Drawing.Font("Segoe UI", 8.25f);
            hint.ForeColor = Color.FromArgb(0xb4, 0xb4, 0xb4);
            hint.SetBounds((int)(24 * k), (int)(108 * k), (int)(392 * k), (int)(20 * k));
            foreach (Label l in new Label[] { head, detail, hint }) { l.AutoEllipsis = true; l.BackColor = BackColor; }
            Controls.Add(head);
            Controls.Add(detail);
            Controls.Add(hint);
            Controls.Add(bar);
        }

        protected override void OnHandleCreated(EventArgs e)
        {
            base.OnHandleCreated(e);
            int on = 1;  // a dark title bar, as the app's (Windows 10 20H1 and later; else it stays light)
            try { DwmSetWindowAttribute(Handle, 20, ref on, 4); } catch (Exception) { }
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            if (!closing) cancelled = true;  // closed by the person: First start stops
            base.OnFormClosing(e);
        }

        void Later(MethodInvoker action)
        {
            try { if (IsHandleCreated && !IsDisposed) BeginInvoke(action); } catch (Exception) { }
        }

        public void Unpacking(long done, long total)
        {
            Later(delegate
            {
                detail.Text = "First start: unpacking the app, " + (done / 1e9).ToString("0.0", CultureInfo.InvariantCulture) +
                              " of " + (total / 1e9).ToString("0.0", CultureInfo.InvariantCulture) + " GB";
                bar.Fraction = total > 0 ? (double)done / total : 0;
            });
        }

        public void Opening()
        {
            Later(delegate
            {
                detail.Text = "Opening the First start window\u2026";
                bar.Fraction = -1;
            });
        }

        public void Finish()
        {
            Later(delegate { closing = true; Close(); });
        }
    }

    // A thin progress bar in the app's colours; Fraction < 0: a segment that moves (no numbers).
    sealed class Bar : Control
    {
        double fraction = -1;
        int phase;
        readonly System.Windows.Forms.Timer timer = new System.Windows.Forms.Timer();

        public Bar()
        {
            SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.OptimizedDoubleBuffer | ControlStyles.UserPaint, true);
            timer.Interval = 30;
            timer.Tick += delegate { phase = (phase + 1) % 100; Invalidate(); };
            timer.Start();
        }

        public double Fraction
        {
            set { fraction = value; timer.Enabled = value < 0; Invalidate(); }
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            using (SolidBrush track = new SolidBrush(Color.FromArgb(0x2a, 0x2a, 0x2a)),
                              fill = new SolidBrush(Color.FromArgb(0x00, 0x90, 0xff)))
            {
                e.Graphics.FillRectangle(track, ClientRectangle);
                int w = Width;
                if (fraction >= 0)
                    e.Graphics.FillRectangle(fill, 0, 0, (int)(w * Math.Min(1, fraction)), Height);
                else
                {
                    int seg = w * 3 / 10, x = (int)((w + seg) * phase / 100.0) - seg;
                    e.Graphics.FillRectangle(fill, x, 0, seg, Height);
                }
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing) timer.Dispose();
            base.Dispose(disposing);
        }
    }

    // ------------------------------------------------------------------ the app

    // The process environment of the app (and so of the stages it starts): what `./clausa app`
    // sets up (runner/envs.py), with the ASR environment's folders and Windows' own on PATH.
    static Dictionary<string, string> AppEnv()
    {
        string env = Path.Combine(Root, "envs", "qwen3-asr");
        string sys = Environment.SystemDirectory, win = Path.GetDirectoryName(sys);
        Dictionary<string, string> e = new Dictionary<string, string>();
        e["CLAUSA_ENVS_DIR"] = Path.Combine(Root, "envs");
        e["CLAUSA_CACHE"] = Path.Combine(Root, "cache");
        e["CLAUSA_LOGS"] = Path.Combine(Root, "logs");
        e["CONDA_PREFIX"] = env;
        e["PATH"] = string.Join(";", new string[] {
            env, Path.Combine(env, "Library", "mingw-w64", "bin"), Path.Combine(env, "Library", "usr", "bin"),
            Path.Combine(env, "Library", "bin"), Path.Combine(env, "Scripts"), Path.Combine(env, "bin"),
            sys, win, Path.Combine(sys, "Wbem"), Path.Combine(sys, "WindowsPowerShell", "v1.0") });
        e["PYTHONNOUSERSITE"] = "1";  // packages a user installed for their own Python 3.12 stay out
        // MFA works in the temp folder: where its path has characters the ANSI code page lacks (a user
        // name in another script), Kaldi could not open its files there; a temp folder inside Clausa's.
        if (NotInAnsiCodePage(Path.GetTempPath()) != null)
        {
            string tmp = Path.Combine(Root, "tmp");
            Directory.CreateDirectory(tmp);
            e["TEMP"] = tmp;
            e["TMP"] = tmp;
        }
        e["PYTHONIOENCODING"] = "utf-8";
        foreach (string v in new string[] { "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE",
                                            "CONDA_DEFAULT_ENV", "CONDA_SHLVL", "CONDA_EXE", "MAMBA_EXE" })
            e[v] = null;
        return e;
    }

    static int StartApp()
    {
        string python = Path.Combine(Root, "envs", "qwen3-asr", "python.exe");
        ProcessStartInfo psi = new ProcessStartInfo(python, "-m runner.app");
        psi.WorkingDirectory = Path.Combine(Root, "app");
        psi.UseShellExecute = false;
        psi.CreateNoWindow = true;
        psi.RedirectStandardOutput = true;
        psi.RedirectStandardError = true;
        psi.StandardOutputEncoding = Encoding.UTF8;
        psi.StandardErrorEncoding = Encoding.UTF8;
        Apply(psi, AppEnv());
        Log("starting Clausa from " + Root);
        // The app keeps its own log (logs\app-<date>.log and each session's log.txt); the console
        // output, which repeats it, is kept for the last start only, for crashes.
        using (StreamWriter console = new StreamWriter(Path.Combine(Root, "logs", "console.log"), false, new UTF8Encoding(false)))
        {
            console.AutoFlush = true;
            object gate = new object();
            DataReceivedEventHandler copy = delegate(object sender, DataReceivedEventArgs a)
            {
                if (a.Data != null) lock (gate) console.WriteLine(a.Data);
            };
            Process p = new Process();
            p.StartInfo = psi;
            p.OutputDataReceived += copy;
            p.ErrorDataReceived += copy;
            p.Start();
            p.BeginOutputReadLine();
            p.BeginErrorReadLine();
            p.WaitForExit();
            int code = p.ExitCode;
            Log("Clausa ended with " + code);
            if (code != 0)
                Message("Clausa closed because of an error. Details are in logs\\console.log in the Clausa folder; " +
                        "please send that file (and logs\\app-<date>.log) with a description of the problem.", true);
            return code;
        }
    }

    // The app's window, the First start window, its console or the Preparing window of this folder's Clausa.
    static bool BringToFront()
    {
        string python = Path.Combine(Root, "envs", "qwen3-asr", "python.exe").ToLowerInvariant();
        string self = Application.ExecutablePath.ToLowerInvariant();
        IntPtr found = IntPtr.Zero;
        bool consoleFallback = File.Exists(ConsoleHandlePath);
        try
        {
            IntPtr console = new IntPtr(long.Parse(File.ReadAllText(ConsoleHandlePath).Trim(), CultureInfo.InvariantCulture));
            if (IsWindow(console) && IsWindowVisible(console) && ClassOf(console) == "ConsoleWindowClass")
                found = console;
        }
        catch (Exception) { }
        if (found == IntPtr.Zero) EnumWindows(delegate(IntPtr hwnd, IntPtr lParam)
        {
            if (!IsWindowVisible(hwnd)) return true;
            StringBuilder text = new StringBuilder(64);
            GetWindowText(hwnd, text, text.Capacity);
            if (!string.Equals(text.ToString(), Title, StringComparison.OrdinalIgnoreCase) &&
                !string.Equals(text.ToString(), Title + ": First start", StringComparison.OrdinalIgnoreCase)) return true;
            if (consoleFallback && string.Equals(text.ToString(), Title + ": First start", StringComparison.OrdinalIgnoreCase))
            {
                found = hwnd;  // the console, hosted by conhost or Windows Terminal
                return false;
            }
            uint pid;
            GetWindowThreadProcessId(hwnd, out pid);
            try
            {
                string exe = Process.GetProcessById((int)pid).MainModule.FileName.ToLowerInvariant();
                if (exe == python || exe == self)
                {
                    found = hwnd;
                    return false;
                }
            }
            catch (Exception) { }
            return true;
        }, IntPtr.Zero);
        if (found == IntPtr.Zero) return false;
        if (IsIconic(found)) ShowWindow(found, 9);  // SW_RESTORE
        SetForegroundWindow(found);
        return true;
    }

    // ------------------------------------------------------------------ helpers

    static int Exec(string exe, string args, string dir, Dictionary<string, string> env, bool console)
    {
        ProcessStartInfo psi = new ProcessStartInfo(exe, args);
        psi.WorkingDirectory = dir;
        psi.UseShellExecute = false;
        psi.CreateNoWindow = !console;
        if (Testing)  // no console of our own: the output comes through ours
        {
            psi.RedirectStandardOutput = psi.RedirectStandardError = true;
            psi.StandardOutputEncoding = psi.StandardErrorEncoding = Encoding.UTF8;
        }
        if (env != null) Apply(psi, env);
        Log("run: " + exe + " " + args);
        using (Process p = new Process())
        {
            p.StartInfo = psi;
            if (Testing)
            {
                DataReceivedEventHandler copy = delegate(object sender, DataReceivedEventArgs a)
                {
                    if (a.Data != null) lock (psi) Console.WriteLine(a.Data);
                };
                p.OutputDataReceived += copy;
                p.ErrorDataReceived += copy;
            }
            p.Start();
            if (Testing)
            {
                p.BeginOutputReadLine();
                p.BeginErrorReadLine();
            }
            p.WaitForExit();
            return p.ExitCode;
        }
    }

    static void Apply(ProcessStartInfo psi, Dictionary<string, string> env)
    {
        foreach (KeyValuePair<string, string> kv in env)
        {
            if (kv.Value == null) psi.EnvironmentVariables.Remove(kv.Key);
            else psi.EnvironmentVariables[kv.Key] = kv.Value;
        }
    }

    // One command-line argument: a backslash before the closing quote is doubled (C:\ -> "C:\\").
    static string Quote(string s)
    {
        return "\"" + (s.EndsWith("\\") ? s + "\\" : s) + "\"";
    }

    static string Gb(long bytes, bool up)
    {
        double gb = bytes / 1e9;
        return (up ? Math.Ceiling(gb) : Math.Floor(gb)).ToString("0");
    }

    static void Message(string text, bool warn)
    {
        if (Testing)
        {
            Console.WriteLine("MESSAGE: " + text);
            return;
        }
        MessageBox.Show(text, Title, MessageBoxButtons.OK, warn ? MessageBoxIcon.Warning : MessageBoxIcon.Information);
    }

    static void Log(string line)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(LogPath));
            File.AppendAllText(LogPath, DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss") + " " + line + "\r\n", Encoding.UTF8);
        }
        catch (Exception) { }
    }

    static void AppendFirstStartLog(string line)
    {
        try
        {
            File.AppendAllText(Path.Combine(Root, "logs", "first-start.log"),
                               DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss") + " " + line + "\r\n", Encoding.UTF8);
        }
        catch (Exception) { }
    }
}
