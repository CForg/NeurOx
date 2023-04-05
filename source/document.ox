#include <oxstd.oxh>

#import <packages/oxdoc/oxdoc_parser>
#import <packages/oxdoc/oxdoc_printer>

///////////////////////////////////////////////////////////////////////
main()
{
    decl p = new OxDoc_Parser(), time = timer();

    // the file to document
    p.Scan("NeurOx.ox").Compile();                 // compile myself
    // pass it to the printer
    decl run = new OxDoc_Printer(p.GetGlobals());
    // set printer options and run it
    run.SetProject("NeurOx:  Neural Networks in Ox")                            // project name
        .SetOutput("../docs")                          // destination of output
        .Exclude("scanner.oxh").Exclude("parser.oxh")
        .Option("latex")                               // uses LaTeX
//        .Option("html_parts/overview.html")            // add this to index.html
        .Run();                                        // Finally, run it
    
    println("time", timespan(time));
}