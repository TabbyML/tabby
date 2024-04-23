-- We start a GraphViz graph
SELECT '
digraph structs {
'
UNION ALL

-- Normally, GraphViz' "dot" command lays out a hierarchical graph from
-- top to bottom.  However, we aren't just laying out individual nodes,
-- each node is a vertical list of database fields.  To prevent GraphViz
-- from snaking arrows all over the place, we constrain it to draw
-- incoming references on the left of each field, and outgoing references
-- on the right.  Since that's the way references flow for each database
-- table, we tell GraphViz to lay the whole graph out left-to-right,
-- which makes its job much easier and produces prettier output.
SELECT '
rankdir="LR"
'
UNION ALL


-- By default, nodes have circles around them.  We will draw our own
-- tables below, we do not want the circles.
SELECT '
node [shape=none]
'
UNION ALL

-- This is the big query that renders a node complete with field names
-- for each table in the database.  Because we want raw GraphViz output,
-- our query returns rows with a single string field, whose value is a
-- complex calculation using SQL as a templating engine.  This is kind
-- of an abuse, but works nicely nevertheless.
SELECT
    CASE
        -- When the previous row's table name is the same as this one,
        -- do nothing.
        WHEN LAG(t.name, 1) OVER (ORDER BY t.name) = t.name THEN ''

        -- Otherwise, this is the first row of a new table, so start
        -- the node markup and add a header row.  Normally in GraphViz,
        -- the table name would *be* the label of the node, but since
        -- we're using the label to represent the entire node, we have
        -- to make our own header.
        --
        -- GraphViz does have a "record" label shape, but it seems tricky
        -- to work with and I found the HTML-style label markup easier
        -- to get working the way I wanted.
        ELSE
            t.name || ' [label=<
            <TABLE BORDER="0" CELLSPACING="0" CELLBORDER="1">
                <TR>
                    <TD COLSPAN="2"><B>' || t.name || '</B></TD>
                </TR>
            '

    -- After the header (if needed), we have rows for each field in
    -- the table.
    --
    -- The "pk" metadata field is zero for table fields that are not part
    -- of the primary key.  If the "pk" metadata field is 1 or more, that
    -- tells you that table field's order in the (potentially composite)
    -- primary key.
    --
    -- We also add ports to each of the table cells, so that we can
    -- later tell GraphViz to specifically connect the ports representing
    -- specific fields in each table, instead of connecting the tables
    -- generally.
    END || '
                <TR>
                    <TD PORT="' || i.name || '_to">' ||
                        CASE i.pk WHEN 0 THEN '&nbsp;' ELSE '🔑' END ||
                    '</TD>
                    <TD PORT="' || i.name || '_from">' || i.name || '</TD>
                </TR>
            ' ||
    CASE
        -- When the next row's table name is the same as this one,
        -- do nothing.
        WHEN LEAD(t.name, 1) OVER (ORDER BY t.name) = t.name THEN ''

        -- Otherwise, this is the last row of a database table, so end
        -- the table markup.
        ELSE '
            </TABLE>
        >];
        '
    END

-- This is how you get nice relational data out of SQLite's metadata
-- pragmas.
FROM pragma_table_list() AS t
    JOIN pragma_table_info(t.name, t.schema) AS i

WHERE
    -- SQLite has a bunch of metadata tables in each schema, which
    -- are hidden from .tables and .schema but which are reported
    -- in pragma_table_list().  They're not user-created and almost
    -- certainly user databases don't have foreign keys to them, so
    -- let's just filter them out.
    t.name NOT LIKE 'sqlite_%'

    -- Despite its name, pragma_table_list() also includes views.
    -- Since those don't store any information or have any correctness
    -- constraints, they're just distracting if you're trying to quickly
    -- understand a database's schema, so we'll filter them out too.
    AND t.type = 'table'
UNION ALL

-- Now we have all the database tables set up, we can draw the links
-- between them.  SQLite gives us the pragma_foreign_key_list() function
-- which (for a given source table) lists all the source fields that are
-- part of a foreign key reference, the target table they refer to, and
-- (if it was created with "REFERENCES table_name(column_name)" syntax,
-- the target column names too.  Unfortunately, if the reference was
-- created with "REFERENCES table_name" syntax, the pragma does *not*
-- figure out what the corresponding target fields are, so we'll also need
-- pragma_table_info() to look up the primary key(s) for the target table.
--
-- Once we have everything we need, we just do a bit more string
-- concatenation to build up the GraphViz syntax equivalent.
--
-- Note that we use the ports we defined above, as well as the directional
-- overrides :e and :w, to force GraphViz to give us a layout that's
-- likely to be readable.
SELECT

    -- We left-join every foreign key field against pragma_table_info
    -- looking for primary keys, and the target table may have a composite
    -- primary key even if the foreign key does not reference the primary
    -- key, so we may wind up with multiple results describing the same
    -- foreign key reference.  DISTINCT makes sure we only describe each
    -- reference once.
    DISTINCT

    t.name || ':' || f."from" || '_from:e -> ' ||

    -- If the constraint was created with "REFERENCES
    -- table_name(column_name)", then f.to will contain 'column_name'.
    -- Otherwise, f.to is NULL, and we need to grab the corresponding
    -- field from the primary key in i.name.
    f."table" || ':' || COALESCE(f."to", i.name) || '_to:w'

FROM pragma_table_list() AS t
    JOIN pragma_foreign_key_list(t.name, t.schema) AS f

    -- We look up all the fields in the target table, just in case
    -- pragma_foreign_key_list() doesn't tell us what the target field
    -- name is.  SQLite doesn't allow foreign-key references to cross
    -- schemas, so it's OK to use the source table's schema name to look
    -- up the target table.
    --
    -- Strictly speaking, we shouldn't need to LEFT JOIN here, a basic
    -- JOIN should do.  This works around a bug in SQLite 3.16.0 to
    -- version 3.45.1: https://sqlite.org/forum/forumpost/b1656fcb39
    LEFT JOIN pragma_table_info(f."table", t.schema) AS i

-- f.seq represents the order of fields in a source table's composite foreign key
-- reference, starting at 0.  In "FOREIGN KEY (a, b)", "a" would have
-- seq=0 and "b" would have seq=1.  i.pk represents the order of fields
-- in a primary key, where "0" means "not part of the primary key".
-- In "PRIMARY KEY (a, b)", "a" would have pk=1 and "b" would have pk=2.
-- For a foreign key reference that specifies the target field name,
-- none of this matters, but if the target field name is missing, then
-- this makes sure that each field of the foreign key reference is joined
-- with the corresponding primary key field of the target table.
WHERE f.seq + 1 = i.pk

UNION ALL

-- Lastly, we close the GraphViz graph.
SELECT '
}';
