set schema 'ml2';

drop table if exists labelled_treatments_seed;
create table labelled_treatments_seed (
  condition_acid varchar(40)
  ,treatment_acid varchar(40)
  ,label integer
);

-- value of 2 = too broad or partial treatment term
-- will not be displayed in app

INSERT INTO ml2.labelled_treatments_seed(condition_acid, treatment_acid, label)
	VALUES 
	('%', '30892', 0)
	-- ('%', '2531', 0)
	-- ,('%', '403118', 0)
	-- ,('%', '457249', 0)
	-- ,('%', '56367', 0)
	-- ,('%', '707002', 0)
	-- ,('%', '127638', 0)
	-- ,('%', '11758', 0)
	-- ,('%', '341471', 0)
	-- ,('%', '36214', 0)
	-- ,('%', '235032', 0)
	-- ,('%', '228687', 0)
	-- ,('%', '509043', 0)
	-- ,('%', '723163', 0)
	-- ,('%', '8126', 0)
	-- ,('%', '217072', 0)
	-- ,('154355', '336184', 0)
	-- ,('112031', '38869', 0)
	-- ,('112031', '326523', 0)
	-- ,('%', '428053', 2)
	-- ,('112031', '507464', 0)
	-- ,('%', '69109', 2)
	-- ,('112031', '436846', 0)
	-- ,('138688', '366023', 0)
	-- ,('44839', '366023', 0)
	-- ,('44839', '106489', 0)
	-- ,('154355', '174980', 0)
	-- ,('58782', '1709', 0)
	-- ,('10609', '75715', 0)
	-- ,('10609', '260685', 0)
	-- ,('10609', '766240', 0)
	-- ,('10609', '6190', 0)
	-- ,('10609', '298347', 0)
	-- ,('10609', '237922', 0)
	-- ,('10609', '182072', 0)
	-- ,('10609', '635415', 1)
	-- ,('10609', '334817', 1)
	-- ,('10609', '27576', 0) 
	-- ,('10609', '67951', 0)
	-- ,('10609', '531083', 0)
	-- ,('10609', '226477', 0) 
	-- ,('10609', '152972', 0)
	-- ,('10609', '200936', 0) 
	-- ,('10609', '52764', 0)  
	-- ,('10609', '25252', 0) 
	-- ,('10609', '687036', 0)
	-- ,('10609', '254012', 0)
	-- ,('10609', '8291', 0) 
	-- ,('10609', '16147', 0)
	-- ,('10609', '285669', 0) 
	-- ,('58782', '137381', 0) 
	-- ,('58782', '182072', 1) 
	-- ,('58782', '486159', 0)   
	-- ,('%', '21416', 0)
	-- ,('%', '213104', 0)
	-- ,('%', '250848', 0)
	-- ,('%', '132114', 0)
	-- ,('125792', '267944', 0)
	-- ,('160029', '134856', 0)
	-- ,('90950', '139033', 1)
	-- ,('90950', '18334', 1) 
	-- ,('154355', '823230', 0)
	-- ,('154355', '37258', 1)
	-- ,('154355', '813124', 1)
	-- ,('154355', '103624', 1)
	-- ,('154355', '101909', 1)
	-- ,('%', '889085', 0)
	-- ,('%', '889251', 0)
	-- ,('%', '889255', 0)
	-- ,('%', '889263', 0)
	;
