* HOTKEY
  * UNIX world, F2 in editor will call out hotkey setup window
  * Ctrl+F mark lines in editor and Ctrl+H unmark
  * Ctrl+Y is to type command in table window
  * Autoexec can run sas code when open sas session
  * R in AIX UNIX editor can copy the line r\ will copy many lines
  * D in AIX UNIX editor delete line


* LIBRARY
  * Copy from one to one and only copy dataset no other marco things
    `Proc copy in=libraryfrom out=libraryout memtype=data;run;`
  * Delete all datasets in one library
    `Proc datasets lib=libraryname kill;run;`
  * List all filese in a directory/library over 360 days
    ```
    Filename dirlist pipe ‘find . -mtime +360’;
    Data dirlist;
    Infile dirlist length=reclen;
    Input filename $varying100. reclen;
    Dpia1 = scan(filename,-2,’.’);
    Dpia2 = scan(filename,-1,’_’);
    run;
    ```

* JANUS
  * Open janus log
  `F11` to open the log window, and open editor.

  Run these code to get log for WorkFlowTool
  `Proc Printto log=log;run;`
  `Options symbolgen macrogen mprint;`
