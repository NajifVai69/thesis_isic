require "core/student_info.lua";

-- STUDENT ONE --
function student_1_id()
    tex.sprint(student_1["id"])
end
function student_1_name()
    tex.sprint(student_1["name"])
end

-- STUDENT TWO --
function student_2_id()
    tex.sprint(student_2["id"])
end
function student_2_name()
    tex.sprint(student_2["name"])
end

-- STUDENT THREE --
function student_3_id()
    tex.sprint(student_3["id"])
end
function student_3_name()
    tex.sprint(student_3["name"])
end

-- STUDENT FOUR --
function student_4_id()
    tex.sprint(student_4["id"])
end
function student_4_name()
    tex.sprint(student_4["name"])
end

-- STUDENT FIVE --
function student_5_id()
    tex.sprint(student_5["id"])
end
function student_5_name()
    tex.sprint(student_5["name"])
end

-- COLORS
function student_1_color()
    if number_of_students > 0 then
        tex.sprint("black")
    else
        tex.sprint("white")
    end
end
function student_2_color()
    if number_of_students > 1 then
        tex.sprint("black")
    else
        tex.sprint("white")
    end
end
function student_3_color()
    if number_of_students > 2 then
        tex.sprint("black")
    else
        tex.sprint("white")
    end
end
function student_4_color()
    if number_of_students > 3 then
        tex.sprint("black")
    else
        tex.sprint("white")
    end
end
function student_5_color()
    if number_of_students > 4 then
        tex.sprint("black")
    else
        tex.sprint("white")
    end
end