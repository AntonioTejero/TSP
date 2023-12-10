classdef City < handle
  
  properties
    id = []
    location = []
  end
  
  events
    
  end
   
  methods (Access = public)

    function city = City(locationX, locationY)
      persistent lastID;
      if isempty(lastID)
        lastID = 0;
      end
      lastID = lastID + 1;
      city.id = lastID;
      city.location = [locationX; locationY];
    end

  end

end